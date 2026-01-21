# train_grpo.py

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import CONFIG


@dataclass(frozen=True)
class QAItem:
    """Single QA item for GRPO training.

    Args:
        question: The user question.
        answer: Ground-truth answer string (GSM8K format).
    """

    question: str
    answer: str


def _build_bnb_config(*, use_4bit: bool) -> BitsAndBytesConfig | None:
    """Create a BitsAndBytes quantization config.

    Args:
        use_4bit: Whether to enable 4-bit NF4 quantization.

    Returns:
        A BitsAndBytesConfig if enabled, else None.
    """
    if not use_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _parse_gsm8k_answer(*, answer_field: str) -> tuple[str, str]:
    """Parse GSM8K 'answer' field into (reasoning, final_answer).

    Args:
        answer_field: Raw GSM8K answer string.

    Returns:
        A tuple (reasoning, final_answer). If parsing fails, final_answer is "".
    """
    if "####" in answer_field:
        parts: list[str] = answer_field.split("####", maxsplit=1)
        reasoning: str = parts[0].strip()
        final_answer: str = parts[1].strip()
        return (reasoning, final_answer)

    return (answer_field.strip(), "")


def _extract_last_alnum(*, text: str) -> str:
    """Extract the last alphanumeric character from a string.

    Args:
        text: Input text.

    Returns:
        The last [A-Za-z0-9] character, or "?" if none is found.
    """
    matches: list[str] = re.findall(pattern=r"[A-Za-z0-9]", string=text)
    if len(matches) == 0:
        return "?"
    return matches[-1]


def _render_chat_prompt(*, prompt: str, cfg: CONFIG, tokenizer: Any) -> str:
    """Render a chat prompt using the model's chat template and system prompt.

    Args:
        prompt: User prompt.
        cfg: Shared configuration.
        tokenizer: Tokenizer providing apply_chat_template().

    Returns:
        Rendered prompt string.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": prompt},
    ]
    rendered: str = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered


def reward_function(*, generated_text: str, ground_truth_answer: str) -> float:
    """Binary reward: compare last alnum of generation vs last alnum of GT answer.

    This is intentionally minimal and robust: it ignores formatting and only
    checks the final alphanumeric character (digit/letter).

    Args:
        generated_text: Model completion text.
        ground_truth_answer: GSM8K raw answer field.

    Returns:
        1.0 if correct, else 0.0.
    """
    _reasoning: str
    gt_final: str
    _reasoning, gt_final = _parse_gsm8k_answer(answer_field=ground_truth_answer)
    gt_last: str = _extract_last_alnum(text=gt_final if gt_final else ground_truth_answer)

    pred_last: str = _extract_last_alnum(text=generated_text)
    return 1.0 if pred_last == gt_last else 0.0


def _compute_sequence_logprob(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Compute summed log-probability of generated tokens only.

    We sum log p(token_t | tokens_<t) over positions belonging to the generated
    continuation (i.e., tokens with index >= prompt_len).

    Args:
        model: Causal LM.
        input_ids: Token ids of shape (1, seq_len).
        attention_mask: Attention mask of shape (1, seq_len).
        prompt_len: Number of tokens in the prompt (prefix) inside input_ids.

    Returns:
        Scalar tensor containing summed log-prob of generated tokens.
    """
    outputs: Any = model(input_ids=input_ids, attention_mask=attention_mask)
    logits: torch.Tensor = outputs.logits  # (1, seq_len, vocab)

    # Next-token prediction alignment:
    # logits[:, t, :] predicts token at position t+1
    shift_logits: torch.Tensor = logits[:, :-1, :]
    shift_labels: torch.Tensor = input_ids[:, 1:]

    log_probs: torch.Tensor = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs: torch.Tensor = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # (1, seq_len-1)

    # Generated tokens are positions >= prompt_len in the original sequence.
    # In shift_labels, position i corresponds to original token at i+1.
    # So we want original positions (prompt_len .. seq_len-1) -> shift positions
    # (prompt_len-1 .. seq_len-2), but excluding any prompt tokens.
    start: int = max(0, int(prompt_len) - 1)
    mask: torch.Tensor = torch.zeros_like(token_log_probs, dtype=torch.bool)
    mask[:, start:] = True

    # Also respect attention mask (shifted)
    shift_attn: torch.Tensor = attention_mask[:, 1:].to(dtype=torch.bool)
    mask = mask & shift_attn

    summed: torch.Tensor = token_log_probs.masked_select(mask).sum()
    return summed


def _load_gsm8k_items(*, cfg: CONFIG) -> list[QAItem]:
    """Load GSM8K train split into QAItem list.

    Args:
        cfg: Shared configuration.

    Returns:
        List of QAItem.
    """
    raw: Any = load_dataset(path=cfg.dataset_name, name=cfg.dataset_config)
    ds: Dataset = raw["train"]
    items: list[QAItem] = [
        QAItem(question=str(ex["question"]), answer=str(ex["answer"])) for ex in ds
    ]
    return items


def _collate_batch(batch: list[QAItem]) -> tuple[list[str], list[str]]:
    """Collate QAItem into two string lists.

    Args:
        batch: List of QAItem.

    Returns:
        (questions, answers) as lists of strings.
    """
    questions: list[str] = [x.question for x in batch]
    answers: list[str] = [x.answer for x in batch]
    return (questions, answers)


def train_grpo() -> None:
    """Train a LoRA policy with a minimal GRPO-style objective.

    Implementation notes (kept minimal and explicit):
    - For each question, sample N=cfg.grpo_group_size completions.
    - Compute rewards per completion using reward_function().
    - Compute group-relative advantages by normalizing rewards within the group.
    - Update policy with REINFORCE-style loss: loss = -adv * logprob(completion).

    This script assumes:
    - You already have an SFT LoRA adapter directory at cfg.output_dir (SFT stage).
    - You want to continue training that adapter with GRPO updates and save to
      cfg.output_dir_grpo (or a fixed path below if not present in config).
    """
    cfg: CONFIG = CONFIG()

    # Ensure BF16 path (avoid FP16 GradScaler).
    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"

    # Resolve paths (minimal: fall back to old constants if not in CONFIG)
    sft_adapter_path: Path = getattr(cfg, "sft_adapter_path", cfg.output_dir)
    output_dir: Path = getattr(cfg, "output_dir_grpo", Path("./weights/final_rlm_lora"))

    group_size_any: Any = getattr(cfg, "grpo_group_size", None)
    group_size: int = int(group_size_any) if group_size_any is not None else 4

    temperature_any: Any = getattr(cfg, "temperature", None)
    top_p_any: Any = getattr(cfg, "top_p", None)
    temperature: float = float(temperature_any) if temperature_any is not None else 0.7
    top_p: float = float(top_p_any) if top_p_any is not None else 0.9

    lr_any: Any = getattr(cfg, "lr_grpo", None)
    lr: float = float(lr_any) if lr_any is not None else float(cfg.lr) * 0.1

    epochs_any: Any = getattr(cfg, "epochs_grpo", None)
    epochs: int = int(epochs_any) if epochs_any is not None else int(cfg.epochs)

    batch_size_any: Any = getattr(cfg, "batch_size_questions_grpo", None)
    batch_size_questions: int = (
        int(batch_size_any) if batch_size_any is not None else int(cfg.batch_size_questions)
    )

    # 1) Tokenizer
    tokenizer: Any = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) Base model (optionally 4-bit) + load SFT adapter as trainable
    bnb_config: BitsAndBytesConfig | None = _build_bnb_config(use_4bit=cfg.use_4bit)

    base_model: Any = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    base_model.generation_config.eos_token_id = tokenizer.eos_token_id

    model: Any = PeftModel.from_pretrained(
        model=base_model,
        model_id=str(sft_adapter_path),
        is_trainable=True,
        local_files_only=True,
    )
    model.train()

    # 3) Optimizer (LoRA params only are trainable)
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
    )

    # 4) Data
    items: list[QAItem] = _load_gsm8k_items(cfg=cfg)
    loader: DataLoader[tuple[list[str], list[str]]] = DataLoader(
        dataset=items,
        batch_size=batch_size_questions,
        shuffle=True,
        collate_fn=_collate_batch,
        drop_last=False,
    )

    # 5) Training loop
    device: torch.device = next(model.parameters()).device
    max_new_tokens: int = int(cfg.max_new_tokens)

    for epoch in range(epochs):
        for questions, answers in loader:
            optimizer.zero_grad(set_to_none=True)

            batch_losses: list[torch.Tensor] = []

            for q, gt in zip(questions, answers, strict=True):
                prompt_text: str = _render_chat_prompt(
                    prompt=q,
                    cfg=cfg,
                    tokenizer=tokenizer,
                )
                enc_prompt: Any = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(cfg.max_seq_len),
                )
                prompt_input_ids: torch.Tensor = enc_prompt["input_ids"].to(device=device)
                prompt_attn: torch.Tensor = enc_prompt["attention_mask"].to(device=device)
                prompt_len: int = int(prompt_input_ids.shape[1])

                # --- Sampling: N completions for this question ---
                with torch.no_grad():
                    gen_out: torch.Tensor = model.generate(
                        input_ids=prompt_input_ids,
                        attention_mask=prompt_attn,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=group_size,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=int(tokenizer.pad_token_id),
                        eos_token_id=int(tokenizer.eos_token_id),
                    )

                # gen_out: (group_size, prompt_len + gen_len_var)
                rewards_list: list[float] = []
                seq_logprobs: list[torch.Tensor] = []

                # For each sampled sequence, compute reward and logprob (with grad)
                for s in range(int(gen_out.shape[0])):
                    seq_ids: torch.Tensor = gen_out[s : s + 1, :]
                    seq_attn: torch.Tensor = (seq_ids != int(tokenizer.pad_token_id)).to(
                        dtype=torch.long
                    )

                    gen_ids_only: torch.Tensor = seq_ids[:, prompt_len:]
                    gen_text: str = tokenizer.decode(
                        gen_ids_only[0],
                        skip_special_tokens=True,
                    )

                    r: float = reward_function(generated_text=gen_text, ground_truth_answer=gt)
                    rewards_list.append(r)

                    seq_logprob: torch.Tensor = _compute_sequence_logprob(
                        model=model,
                        input_ids=seq_ids,
                        attention_mask=seq_attn.to(device=device),
                        prompt_len=prompt_len,
                    )
                    seq_logprobs.append(seq_logprob)

                rewards: torch.Tensor = torch.tensor(
                    rewards_list,
                    dtype=torch.float32,
                    device=device,
                )

                # --- Group-relative advantage (normalize within the group) ---
                mean_r: torch.Tensor = rewards.mean()
                std_r: torch.Tensor = rewards.std(unbiased=False)
                advantages: torch.Tensor = (rewards - mean_r) / (std_r + 1e-8)

                # --- Loss: -adv * logprob ---
                # Detach advantages so we only backprop through log-probs.
                adv_detached: torch.Tensor = advantages.detach()
                logprob_stack: torch.Tensor = torch.stack(seq_logprobs, dim=0)
                loss_q: torch.Tensor = -(adv_detached * logprob_stack).mean()

                batch_losses.append(loss_q)

            if len(batch_losses) == 0:
                continue

            loss: torch.Tensor = torch.stack(batch_losses, dim=0).mean()
            loss.backward()

            # Basic gradient clip (safe for LoRA)
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)

            optimizer.step()

        # Save after each epoch (minimal + robust)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_directory=str(output_dir))
        tokenizer.save_pretrained(save_directory=str(output_dir))

    print(f"GRPO training finished. Saved adapter to: {output_dir}")


if __name__ == "__main__":
    train_grpo()
