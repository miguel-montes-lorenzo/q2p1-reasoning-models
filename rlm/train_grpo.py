# train_grpo.py

from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path
from typing import Any

import torch
from config import GRPO_CONFIG
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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
        return reasoning, final_answer
    return answer_field.strip(), ""


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


def _extract_tag_content(*, text: str, tag: str) -> str | None:
    """Extract inner content of a tag like <tag>...</tag>.

    Args:
        text: Full model completion.
        tag: Tag name without brackets (e.g., "think", "answer").

    Returns:
        Inner content if found, else None.
    """
    m: re.Match[str] | None = re.search(
        pattern=rf"<{tag}>\s*(.*?)\s*</{tag}>",
        string=text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m is None:
        return None
    return m.group(1).strip()


def _has_think_block(*, text: str) -> bool:
    """Return True if <think>...</think> exists."""
    return (
        re.search(
            pattern=r"<think>\s*.*?\s*</think>",
            string=text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        is not None
    )


def _has_answer_block(*, text: str) -> bool:
    """Return True if <answer>...</answer> exists."""
    return (
        re.search(
            pattern=r"<answer>\s*.*?\s*</answer>",
            string=text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        is not None
    )


def _tags_in_strict_order(*, text: str) -> bool:
    """Return True if tags appear as <think> </think> <answer> </answer> in order."""
    lower: str = text.lower()
    idx_t0: int = lower.find("<think>")
    idx_t1: int = lower.find("</think>")
    idx_a0: int = lower.find("<answer>")
    idx_a1: int = lower.find("</answer>")

    if min(idx_t0, idx_t1, idx_a0, idx_a1) < 0:
        return False
    return idx_t0 < idx_t1 < idx_a0 < idx_a1


def _no_text_outside_tags(*, text: str) -> bool:
    """Return True if only <think>...</think><answer>...</answer> (plus whitespace)."""
    lower: str = text.lower()

    think_block: re.Match[str] | None = re.search(
        pattern=r"<think>\s*.*?\s*</think>",
        string=lower,
        flags=re.DOTALL,
    )
    tmp: str = lower
    if think_block is not None:
        tmp = tmp[: think_block.start()] + tmp[think_block.end() :]

    answer_block: re.Match[str] | None = re.search(
        pattern=r"<answer>\s*.*?\s*</answer>",
        string=tmp,
        flags=re.DOTALL,
    )
    if answer_block is not None:
        tmp = tmp[: answer_block.start()] + tmp[answer_block.end() :]

    return tmp.strip() == ""


def _len_ratio_score(*, think_text: str | None, answer_text: str | None) -> float:
    """Compute len(think) / (len(think) + len(answer)) in [0,1].

    Args:
        think_text: Extracted think content.
        answer_text: Extracted answer content.

    Returns:
        Ratio in [0,1]. Returns 0.0 if missing or denominator is 0.
    """
    if think_text is None or answer_text is None:
        return 0.0
    lt: int = len(think_text)
    la: int = len(answer_text)
    denom: int = lt + la
    if denom <= 0:
        return 0.0
    return float(lt) / float(denom)


def _naturalness_score(*, text: str) -> float:
    """Heuristic naturalness score in [0, 1] for reasoning text.

    This is a fast proxy to penalize low-information padding and repetitions.

    Signals:
      - Compression ratio (zlib): repetitive text compresses better -> lower score.
      - Lexical diversity: unique words / total words.
      - Repetition penalty: common n-gram repetitions and long character runs.

    Args:
        text: The reasoning text (ideally content inside <think>...</think>).

    Returns:
        Score in [0,1].
    """
    s: str = text.strip()
    if not s:
        return 0.0

    b: bytes = s.encode("utf-8", errors="ignore")
    if len(b) == 0:
        return 0.0

    comp: bytes = zlib.compress(b, level=6)
    comp_ratio: float = float(len(comp)) / float(len(b))
    comp_norm: float = (comp_ratio - 0.20) / (1.00 - 0.20)
    comp_norm = max(0.0, min(1.0, comp_norm))

    words: list[str] = re.findall(pattern=r"[A-Za-z0-9_]+", string=s.lower())
    if not words:
        diversity: float = 0.0
    else:
        diversity = float(len(set(words))) / float(len(words))

    max_run: int = 1
    run: int = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 1

    run_penalty: float = 0.0
    if max_run >= 12:
        run_penalty = 0.50
    elif max_run >= 8:
        run_penalty = 0.25
    elif max_run >= 6:
        run_penalty = 0.10

    bigrams: list[tuple[str, str]] = []
    if len(words) >= 2:
        bigrams = list(zip(words[:-1], words[1:], strict=True))

    if not bigrams:
        rep_penalty: float = 0.0
    else:
        counts: dict[tuple[str, str], int] = {}
        for bg in bigrams:
            counts[bg] = counts.get(bg, 0) + 1
        most_common: int = max(counts.values())
        rep_frac: float = float(most_common) / float(len(bigrams))

        rep_penalty = 0.0
        if rep_frac >= 0.35:
            rep_penalty = 0.50
        elif rep_frac >= 0.25:
            rep_penalty = 0.25
        elif rep_frac >= 0.18:
            rep_penalty = 0.10

    base: float = 0.55 * comp_norm + 0.45 * diversity
    score: float = base * (1.0 - run_penalty) * (1.0 - rep_penalty)
    return max(0.0, min(1.0, score))


def _extract_final_int(*, text: str) -> int | None:
    """Extract the last integer substring from text.

    Args:
        text: Input string.

    Returns:
        Last parsed integer if any, else None.
    """
    matches: list[str] = re.findall(pattern=r"[-+]?\d+", string=text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def reward_function(*, generated_text: str, ground_truth_answer: str) -> float:
    """Compute a shaped reward for GSM8K with formatting + naturalness bonuses.

    Rewards:
      +0.50 correct final answer
      +0.25 structure compliance (sum of subparts):
        +0.05 has <think>...</think>
        +0.05 has <answer>...</answer>
        +0.05 no text outside tag scopes
        +0.10 tags appear in strict order:
              <think> -> </think> -> <answer> -> </answer>
      +0.10 length ratio:
        len(think) / (len(think) + len(answer))
      +0.15 naturalness heuristic score on think content

    Args:
        generated_text: Model completion (generated continuation).
        ground_truth_answer: GSM8K ground-truth answer field (with #### marker).

    Returns:
        Total reward in [0, 1] (clamped).
    """
    reward: float = 0.0

    think_text: str | None = _extract_tag_content(text=generated_text, tag="think")
    answer_text: str | None = _extract_tag_content(text=generated_text, tag="answer")

    pred: int | None
    if answer_text is not None:
        pred = _extract_final_int(text=answer_text)
    else:
        pred = _extract_final_int(text=generated_text)

    _, gt_final_str = _parse_gsm8k_answer(answer_field=ground_truth_answer)
    gt: int | None = _extract_final_int(text=gt_final_str)

    if pred is not None and gt is not None and pred == gt:
        reward += 0.50

    has_think: bool = _has_think_block(text=generated_text)
    has_answer: bool = _has_answer_block(text=generated_text)

    if has_think:
        reward += 0.05
    if has_answer:
        reward += 0.05
    if has_think and has_answer and _no_text_outside_tags(text=generated_text):
        reward += 0.05
    if _tags_in_strict_order(text=generated_text):
        reward += 0.10

    reward += 0.10 * _len_ratio_score(think_text=think_text, answer_text=answer_text)

    nat_input: str = think_text if think_text is not None else generated_text
    reward += 0.15 * _naturalness_score(text=nat_input)

    return float(max(0.0, min(1.0, reward)))


def _build_prompt_text(*, tokenizer: Any, cfg: GRPO_CONFIG, question: str) -> str:
    """Build a chat-template prompt for GSM8K question-only training.

    Args:
        tokenizer: HF tokenizer.
        cfg: GRPO configuration.
        question: GSM8K question string.

    Returns:
        Rendered prompt string.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": question},
    ]
    prompt: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return f"{prompt}<think>"


def _safe_std(*, x: torch.Tensor) -> torch.Tensor:
    """Compute std with a safe fallback for small/constant tensors.

    Args:
        x: Input tensor.

    Returns:
        Standard deviation tensor.
    """
    if int(x.numel()) <= 1:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return x.std(unbiased=False)


def _collate_grpo_batch(*, examples: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Collate function producing lists of questions and answers.

    Args:
        examples: Raw dataset examples.

    Returns:
        Dict with 'question' and 'answer' lists.
    """
    questions: list[str] = [str(ex["question"]) for ex in examples]
    answers: list[str] = [str(ex["answer"]) for ex in examples]
    return {"question": questions, "answer": answers}


def _load_train_dataset(*, cfg: GRPO_CONFIG) -> Dataset:
    """Load GSM8K train split, optionally truncate.

    Args:
        cfg: GRPO configuration.

    Returns:
        Train dataset.
    """
    raw: Any = load_dataset(path=cfg.dataset_name, name=cfg.dataset_config)
    ds: Dataset = raw["train"]
    if cfg.max_train_examples is not None and len(ds) > int(cfg.max_train_examples):
        ds = ds.select(range(int(cfg.max_train_examples)))
    return ds


def _count_trainable_params(*, model: torch.nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters.

    Args:
        model: Torch model.

    Returns:
        (trainable_params, total_params).
    """
    trainable: int = 0
    total: int = 0
    for p in model.parameters():
        n: int = int(p.numel())
        total += n
        if bool(p.requires_grad):
            trainable += n
    return trainable, total


def _ensure_only_trainable_params(*, model: torch.nn.Module) -> None:
    """Ensure that only LoRA adapter parameters are trainable.

    Args:
        model: Torch model.
    """
    if not isinstance(model, PeftModel):
        return
    for name, param in model.named_parameters():
        param.requires_grad = bool("lora_" in name)


def _maybe_wrap_with_lora(
    *, base_model: torch.nn.Module, cfg: GRPO_CONFIG
) -> torch.nn.Module:
    """Create a LoRA adapter on top of a base model.

    Args:
        base_model: Base HF model.
        cfg: GRPO configuration.

    Returns:
        PEFT-wrapped model.
    """
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_cfg: LoraConfig = LoraConfig(
        r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(base_model, lora_cfg)


def _decode_generated_only(
    *,
    tokenizer: Any,
    sequences: torch.Tensor,
    prompt_len: int,
) -> list[str]:
    """Decode only the generated continuation (excluding the prompt tokens).

    Args:
        tokenizer: HF tokenizer.
        sequences: Generated token ids [B, T].
        prompt_len: Prompt length in tokens.

    Returns:
        List of decoded continuations per sequence.
    """
    cont_ids: list[torch.Tensor] = []
    bsz: int = int(sequences.shape[0])
    for b in range(bsz):
        seq: torch.Tensor = sequences[b]
        cont_ids.append(
            seq[int(prompt_len) :] if int(prompt_len) < int(seq.numel()) else seq[:0]
        )
    return tokenizer.batch_decode(cont_ids, skip_special_tokens=True)


def _gather_generated_logp_stats(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (sum_logp, n_gen_tokens) for generated tokens per sequence."""
    outputs: Any = model(input_ids=input_ids, attention_mask=attention_mask)
    logits: torch.Tensor = outputs.logits  # [B, T, V]

    log_probs_all: torch.Tensor = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_tokens: torch.Tensor = input_ids[:, 1:]  # [B, T-1]
    attn: torch.Tensor = attention_mask[:, 1:]  # [B, T-1]

    start_tgt: int = max(int(prompt_len) - 1, 0)
    bsz: int = int(input_ids.shape[0])
    sums: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []

    for b in range(bsz):
        gen_mask: torch.Tensor = torch.zeros_like(attn[b], dtype=torch.bool)
        if start_tgt < int(gen_mask.numel()):
            gen_mask[start_tgt:] = True
        gen_mask = gen_mask & attn[b].to(dtype=torch.bool)

        picked: torch.Tensor = (
            log_probs_all[b]
            .gather(
                dim=-1,
                index=target_tokens[b].unsqueeze(-1),
            )
            .squeeze(-1)
        )  # [T-1]

        sums.append(picked[gen_mask].sum())
        counts.append(torch.tensor(int(gen_mask.sum().item()), device=input_ids.device))

    return torch.stack(sums, dim=0), torch.stack(counts, dim=0)


def _parse_step_from_checkpoint_dirname(*, name: str) -> int | None:
    """Parse step number from a directory name like 'checkpoint-step-123'."""
    m: re.Match[str] | None = re.fullmatch(
        pattern=r"checkpoint-step-(\d+)", string=name
    )
    if m is None:
        return None
    return int(m.group(1))


def _rotate_checkpoints(*, output_dir: Path, keep_last: int) -> None:
    """Keep only the most recent N checkpoint directories."""
    if int(keep_last) <= 0 or not output_dir.exists():
        return

    ckpts: list[tuple[int, Path]] = []
    for p in output_dir.iterdir():
        if not p.is_dir():
            continue
        step: int | None = _parse_step_from_checkpoint_dirname(name=p.name)
        if step is not None:
            ckpts.append((step, p))

    ckpts.sort(key=lambda t: t[0])
    to_delete: list[Path] = [p for _s, p in ckpts[:-keep_last]]
    if not to_delete:
        return

    import shutil

    for p in to_delete:
        shutil.rmtree(path=p, ignore_errors=True)


def _save_checkpoint(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    step: int,
    keep_last: int,
) -> None:
    """Save a PEFT adapter checkpoint and rotate older checkpoints."""
    if not isinstance(model, PeftModel):
        raise RuntimeError("Refusing to checkpoint a non-PEFT model.")
    ckpt_dir: Path = output_dir / f"checkpoint-step-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_directory=str(ckpt_dir))
    tokenizer.save_pretrained(save_directory=str(ckpt_dir))
    _rotate_checkpoints(output_dir=output_dir, keep_last=keep_last)


def _load_base_and_policy(
    *,
    cfg: GRPO_CONFIG,
    adapter_path: Path | None,
) -> tuple[torch.nn.Module, torch.nn.Module, Any]:
    """Load (base_model_ref, policy_model_with_lora, tokenizer)."""
    tokenizer: Any = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config: BitsAndBytesConfig | None = _build_bnb_config(use_4bit=cfg.use_4bit)

    base_model: Any = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.config.pad_token_id = int(tokenizer.pad_token_id)
    base_model.generation_config.pad_token_id = int(tokenizer.pad_token_id)
    base_model.generation_config.eos_token_id = int(tokenizer.eos_token_id)

    if bool(cfg.use_4bit):
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=False,
        )
        base_model.config.use_cache = False

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    if adapter_path is None:
        policy_model: torch.nn.Module = _maybe_wrap_with_lora(
            base_model=base_model, cfg=cfg
        )
    else:
        policy_model = PeftModel.from_pretrained(
            model=base_model,
            model_id=str(adapter_path),
            is_trainable=True,
        )

    _ensure_only_trainable_params(model=policy_model)
    return base_model, policy_model, tokenizer


def _get_learning_rate(*, optimizer: torch.optim.Optimizer) -> float:
    """Get the current learning rate from an optimizer."""
    return float(optimizer.param_groups[0]["lr"])


def main(*, adapter_path: Path | None = None) -> None:
    """Run GRPO-style policy optimization with a KL penalty to a reference model."""
    cfg: GRPO_CONFIG = GRPO_CONFIG()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ref_model: torch.nn.Module
    policy_model: torch.nn.Module
    tokenizer: Any
    ref_model, policy_model, tokenizer = _load_base_and_policy(
        cfg=cfg,
        adapter_path=adapter_path,
    )
    policy_model.train()

    trainable_n, total_n = _count_trainable_params(model=policy_model)
    print(
        f"Trainable params: {trainable_n} / {total_n} "
        f"({100.0 * float(trainable_n) / float(total_n):.2f}%)"
    )

    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        params=[p for p in policy_model.parameters() if bool(p.requires_grad)],
        lr=float(cfg.lr),
    )

    ds: Dataset = _load_train_dataset(cfg=cfg)
    loader: DataLoader[dict[str, list[str]]] = DataLoader(
        ds,
        batch_size=int(cfg.batch_size_questions),
        shuffle=True,
        drop_last=False,
        collate_fn=lambda xs: _collate_grpo_batch(examples=xs),
    )

    out_dir: Path = Path(cfg.checkpoint_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    device: torch.device = next(policy_model.parameters()).device

    step: int = 0
    optimizer.zero_grad(set_to_none=True)

    group_size: int = int(cfg.group_size)
    max_new_tokens: int = int(cfg.max_new_tokens)
    temperature: float = float(cfg.temperature)
    top_p: float | None = cfg.top_p
    beta_kl: float = float(cfg.beta_kl)

    logging_interval: int = int(cfg.logging_interval)
    checkpoint_interval: int = int(cfg.checkpoint_interval)
    keep_last: int = int(cfg.keep_last_checkpoints)

    grad_accum_steps: int = int(cfg.grad_accum_steps)
    clip_grad_norm: float = float(cfg.clip_grad_norm)

    for epoch in range(int(cfg.epochs)):
        epoch_loss_sum: float = 0.0
        epoch_loss_count: int = 0

        loss_buf: list[float] = []
        r_mean_buf: list[float] = []
        r_std_buf: list[float] = []
        acc_buf: list[float] = []

        pbar: tqdm[dict[str, list[str]]] = tqdm(
            loader,
            desc=f"epoch {epoch + 1}/{int(cfg.epochs)}",
            dynamic_ncols=True,
        )

        for batch in pbar:
            batch_questions: list[str] = batch["question"]
            batch_answers: list[str] = batch["answer"]

            total_loss: torch.Tensor = torch.tensor(
                0.0, device=device, dtype=torch.float32
            )
            batch_rewards_all: list[float] = []

            for q, gt in zip(batch_questions, batch_answers, strict=True):
                prompt_text: str = _build_prompt_text(
                    tokenizer=tokenizer,
                    cfg=cfg,
                    question=q,
                )
                prompt_inputs: Any = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=int(cfg.max_seq_len),
                )

                prompt_input_ids: torch.Tensor = prompt_inputs["input_ids"].to(
                    device=device
                )
                prompt_attention_mask: torch.Tensor = prompt_inputs[
                    "attention_mask"
                ].to(device=device)
                prompt_len: int = int(prompt_input_ids.shape[1])

                gen_kwargs: dict[str, Any] = {
                    "input_ids": prompt_input_ids,
                    "attention_mask": prompt_attention_mask,
                    "do_sample": True,
                    "temperature": float(temperature),
                    "max_new_tokens": int(max_new_tokens),
                    "num_return_sequences": int(group_size),
                    "pad_token_id": int(tokenizer.pad_token_id),
                    "eos_token_id": int(tokenizer.eos_token_id),
                }
                if top_p is not None:
                    gen_kwargs["top_p"] = float(top_p)

                policy_model.eval()
                with torch.no_grad():
                    gen_ids: torch.Tensor = policy_model.generate(**gen_kwargs)
                policy_model.train()

                seq_lens: torch.Tensor = torch.tensor(
                    [int(s.shape[0]) for s in gen_ids],
                    device=gen_ids.device,
                    dtype=torch.long,
                )
                max_len: int = int(seq_lens.max().item())

                padded_ids: list[torch.Tensor] = []
                padded_masks: list[torch.Tensor] = []
                for s in gen_ids:
                    pad_len: int = int(max_len - int(s.shape[0]))
                    if pad_len > 0:
                        pad_ids: torch.Tensor = torch.full(
                            (pad_len,),
                            fill_value=int(tokenizer.pad_token_id),
                            device=s.device,
                            dtype=s.dtype,
                        )
                        s2: torch.Tensor = torch.cat([s, pad_ids], dim=0)
                        m2: torch.Tensor = torch.cat(
                            [
                                torch.ones_like(s, dtype=torch.long),
                                torch.zeros_like(pad_ids, dtype=torch.long),
                            ],
                            dim=0,
                        )
                    else:
                        s2 = s
                        m2 = torch.ones_like(s, dtype=torch.long)

                    padded_ids.append(s2.unsqueeze(0))
                    padded_masks.append(m2.unsqueeze(0))

                group_input_ids: torch.Tensor = torch.cat(padded_ids, dim=0)
                group_attention_mask: torch.Tensor = torch.cat(padded_masks, dim=0)

                decoded_cont: list[str] = _decode_generated_only(
                    tokenizer=tokenizer,
                    sequences=group_input_ids,
                    prompt_len=prompt_len,
                )
                rewards_list: list[float] = [
                    reward_function(generated_text=t, ground_truth_answer=gt)
                    for t in decoded_cont
                ]
                batch_rewards_all.extend(rewards_list)

                rewards_tensor: torch.Tensor = torch.tensor(
                    rewards_list,
                    device=group_input_ids.device,
                    dtype=torch.float32,
                )
                mean_reward: torch.Tensor = rewards_tensor.mean()
                std_reward: torch.Tensor = _safe_std(x=rewards_tensor)
                advantages: torch.Tensor = (rewards_tensor - mean_reward) / (
                    std_reward + 1e-8
                )
                adv_detached: torch.Tensor = advantages.detach()

                sum_logp_pi, n_gen = _gather_generated_logp_stats(
                    model=policy_model,
                    input_ids=group_input_ids,
                    attention_mask=group_attention_mask,
                    prompt_len=prompt_len,
                )
                with torch.no_grad():
                    sum_logp_ref, _ = _gather_generated_logp_stats(
                        model=ref_model,
                        input_ids=group_input_ids,
                        attention_mask=group_attention_mask,
                        prompt_len=prompt_len,
                    )

                n_gen_safe: torch.Tensor = torch.clamp(
                    n_gen.to(dtype=torch.float32), min=1.0
                )
                mean_logp_pi: torch.Tensor = (
                    sum_logp_pi.to(dtype=torch.float32) / n_gen_safe
                )
                mean_logp_ref: torch.Tensor = (
                    sum_logp_ref.to(dtype=torch.float32) / n_gen_safe
                )

                approx_kl: torch.Tensor = mean_logp_pi - mean_logp_ref
                loss_q: torch.Tensor = -(
                    adv_detached * mean_logp_pi
                ).mean() + beta_kl * (approx_kl.mean())
                total_loss = total_loss + loss_q

            total_loss = total_loss / float(max(len(batch_questions), 1))
            total_loss = total_loss / float(grad_accum_steps)
            total_loss.backward()

            step += 1

            if step % int(grad_accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters=[
                        p for p in policy_model.parameters() if bool(p.requires_grad)
                    ],
                    max_norm=float(clip_grad_norm),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_val: float = float(total_loss.detach().cpu().item())
            epoch_loss_sum += loss_val
            epoch_loss_count += 1

            if len(batch_rewards_all) > 0:
                rewards_step: torch.Tensor = torch.tensor(
                    batch_rewards_all,
                    device=device,
                    dtype=torch.float32,
                )
                reward_mean: float = float(rewards_step.mean().detach().cpu().item())
                reward_std: float = float(
                    _safe_std(x=rewards_step).detach().cpu().item()
                )
                accuracy: float = reward_mean
            else:
                reward_mean = 0.0
                reward_std = 0.0
                accuracy = 0.0

            loss_buf.append(loss_val)
            r_mean_buf.append(reward_mean)
            r_std_buf.append(reward_std)
            acc_buf.append(accuracy)
            if logging_interval > 0:
                loss_buf = loss_buf[-logging_interval:]
                r_mean_buf = r_mean_buf[-logging_interval:]
                r_std_buf = r_std_buf[-logging_interval:]
                acc_buf = acc_buf[-logging_interval:]

            avg_loss: float = float(sum(loss_buf) / max(len(loss_buf), 1))
            avg_r_mean: float = float(sum(r_mean_buf) / max(len(r_mean_buf), 1))
            avg_r_std: float = float(sum(r_std_buf) / max(len(r_std_buf), 1))
            avg_acc: float = float(sum(acc_buf) / max(len(acc_buf), 1))
            lr: float = _get_learning_rate(optimizer=optimizer)

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "r_mean": f"{avg_r_mean:.3f}",
                "acc": f"{avg_acc:.3f}",
                "r_std": f"{avg_r_std:.3f}",
                "lr": f"{lr:.2e}",
            })

            if logging_interval > 0 and step % logging_interval == 0:
                print()
                print(
                    f"[step={step + 1}] "
                    f"loss: {avg_loss:.4f}, "
                    f": {avg_r_mean:.3f}, "
                    f"acc: {avg_acc:.3f}, "
                    f": {avg_r_std:.3f}, "
                    f"lr: {lr:.2e}, "
                )

            if checkpoint_interval > 0 and step % checkpoint_interval == 0:
                _save_checkpoint(
                    model=policy_model,
                    tokenizer=tokenizer,
                    output_dir=out_dir,
                    step=step,
                    keep_last=keep_last,
                )

        epoch_loss_mean: float = epoch_loss_sum / float(max(epoch_loss_count, 1))
        print(f"[epoch={epoch + 1}] epoch_loss_mean={epoch_loss_mean:.4f}")

    if not isinstance(policy_model, PeftModel):
        raise RuntimeError("Training finished but policy_model is not a PEFT model.")

    policy_model.save_pretrained(save_directory=str(out_dir))
    tokenizer.save_pretrained(save_directory=str(out_dir))
    print("GRPO Training finished")


def _parse_optional_checkpoint_arg(*, argv: list[str]) -> Path | None:
    """Parse optional checkpoint path from argv[1]."""
    if len(argv) < 2:
        return None
    raw: str = argv[1].strip()
    return Path(raw) if raw else None


if __name__ == "__main__":
    path: Path | None = _parse_optional_checkpoint_arg(argv=sys.argv)

    default_best: Path = Path("weights/sft_lora/best-checkpoint")
    if path is None and default_best.exists():
        path = default_best

    main(adapter_path=path)
