# train_grpo.py

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import torch
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

from config import GRPO_CONFIG


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

    reasoning_fallback: str = answer_field.strip()
    final_fallback: str = ""
    return (reasoning_fallback, final_fallback)


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


def _extract_final_int(*, text: str) -> int | None:
    """Extract an integer from text by taking the last integer substring.

    Args:
        text: Generated model text.

    Returns:
        Extracted integer if found, else None.
    """
    matches: list[str] = re.findall(pattern=r"[-+]?\d+", string=text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def reward_function(*, generated_text: str, ground_truth_answer: str) -> float:
    """Compute a verifiable {0,1} reward for GSM8K-style answers.

    Note:
        This expects `generated_text` to be ONLY the model's generated continuation
        (not including the prompt), to avoid picking integers from the question.

    Args:
        generated_text: Generated continuation produced by the assistant.
        ground_truth_answer: GSM8K answer field containing rationale + final answer.

    Returns:
        1.0 if extracted final integer matches ground-truth integer, else 0.0.
    """
    pred: int | None = _extract_final_int(text=generated_text)
    _reasoning: str
    gt_final_str: str
    _reasoning, gt_final_str = _parse_gsm8k_answer(answer_field=ground_truth_answer)
    gt: int | None = _extract_final_int(text=gt_final_str)

    if pred is None or gt is None:
        return 0.0
    return 1.0 if pred == gt else 0.0


def _build_prompt_text(*, tokenizer: Any, cfg: GRPO_CONFIG, question: str) -> str:
    """Build a chat-template prompt for GSM8K question-only inference/training.

    Args:
        tokenizer: HF tokenizer implementing apply_chat_template().
        cfg: GRPO configuration.
        question: GSM8K question.

    Returns:
        Prompt text including the assistant generation prompt.
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
    return prompt


def _gather_logprobs_of_generated_tokens(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sequence sum log-probabilities over generated tokens.

    Args:
        model: Causal LM.
        input_ids: Token ids, shape [B, T].
        attention_mask: Attention mask, shape [B, T].
        prompt_lens: Prompt lengths (number of tokens), shape [B].

    Returns:
        Sum log-prob for generated tokens per sequence, shape [B].
    """
    outputs: Any = model(input_ids=input_ids, attention_mask=attention_mask)
    logits: torch.Tensor = outputs.logits  # [B, T, V]

    log_probs_all: torch.Tensor = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_tokens: torch.Tensor = input_ids[:, 1:]  # [B, T-1]

    bsz: int = int(input_ids.shape[0])
    sums: list[torch.Tensor] = []

    for b in range(bsz):
        prompt_len: int = int(prompt_lens[b].item())
        start_tgt: int = max(prompt_len - 1, 0)

        attn: torch.Tensor = attention_mask[b, 1:]  # [T-1]
        gen_mask: torch.Tensor = torch.zeros_like(attn, dtype=torch.bool)
        if start_tgt < int(gen_mask.numel()):
            gen_mask[start_tgt:] = True
        gen_mask = gen_mask & attn.to(dtype=torch.bool)

        lp_row: torch.Tensor = log_probs_all[b]  # [T-1, V]
        tgt_row: torch.Tensor = target_tokens[b]  # [T-1]
        picked: torch.Tensor = lp_row.gather(dim=-1, index=tgt_row.unsqueeze(-1)).squeeze(
            -1
        )
        sums.append(picked[gen_mask].sum())

    return torch.stack(sums, dim=0)


def _collate_grpo_batch(*, examples: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Collate function producing lists of questions and answers.

    Args:
        examples: List of GSM8K dataset rows.

    Returns:
        Dict with "question" and "answer" lists.
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

    max_examples: int | None = cfg.max_train_examples
    if max_examples is not None and int(max_examples) > 0 and len(ds) > int(max_examples):
        ds = ds.select(range(int(max_examples)))
    return ds


def _ensure_only_trainable_params(*, model: torch.nn.Module) -> None:
    """Ensure that only adapter parameters are trainable when using PEFT.

    Args:
        model: Model to validate.
    """
    if not isinstance(model, PeftModel):
        return

    for name, param in model.named_parameters():
        is_lora: bool = "lora_" in name
        param.requires_grad = bool(is_lora)


def _count_trainable_params(*, model: torch.nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters.

    Args:
        model: Model to inspect.

    Returns:
        (trainable, total)
    """
    trainable: int = 0
    total: int = 0
    for p in model.parameters():
        n: int = int(p.numel())
        total += n
        if bool(p.requires_grad):
            trainable += n
    return trainable, total


def _maybe_wrap_with_lora(*, base_model: torch.nn.Module, cfg: GRPO_CONFIG) -> torch.nn.Module:
    """Create a LoRA adapter on top of a base model.

    Note:
        LoRA hyperparameters are read via getattr() so the config can remain
        minimal. If you add these fields to GRPO_CONFIG later, this will pick
        them up automatically.

    Args:
        base_model: Base Causal LM.
        cfg: GRPO configuration.

    Returns:
        A PEFT-wrapped model.
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

    lora_r: int = int(getattr(cfg, "lora_r", 16))
    lora_alpha: int = int(getattr(cfg, "lora_alpha", 32))
    lora_dropout: float = float(getattr(cfg, "lora_dropout", 0.05))

    lora_cfg: LoraConfig = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model: torch.nn.Module = get_peft_model(base_model, lora_cfg)
    return model


def _load_model_and_tokenizer(
    *,
    cfg: GRPO_CONFIG,
    adapter_path: Path | None,
) -> tuple[torch.nn.Module, Any]:
    """Load base model and a trainable LoRA adapter.

    Behavior:
        - If adapter_path is provided: load that adapter as trainable.
        - If adapter_path is None: create a fresh LoRA adapter as trainable.

    Args:
        cfg: GRPO configuration.
        adapter_path: Optional adapter directory.

    Returns:
        (model, tokenizer)
    """
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

    if adapter_path is None:
        model_no_ckpt: torch.nn.Module = _maybe_wrap_with_lora(
            base_model=base_model,
            cfg=cfg,
        )
        _ensure_only_trainable_params(model=model_no_ckpt)
        return model_no_ckpt, tokenizer

    model_ckpt: torch.nn.Module = PeftModel.from_pretrained(
        model=base_model,
        model_id=str(adapter_path),
        is_trainable=True,
    )
    _ensure_only_trainable_params(model=model_ckpt)
    return model_ckpt, tokenizer


def _safe_std(*, x: torch.Tensor) -> torch.Tensor:
    """Compute std with a safe fallback for small/constant tensors.

    Args:
        x: Input tensor.

    Returns:
        Standard deviation tensor (scalar).
    """
    if int(x.numel()) <= 1:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    std: torch.Tensor = x.std(unbiased=False)
    return std


def _get_learning_rate(*, optimizer: torch.optim.Optimizer) -> float:
    """Get the current learning rate from an optimizer.

    Args:
        optimizer: Torch optimizer.

    Returns:
        Learning rate of the first parameter group.
    """
    lr: float = float(optimizer.param_groups[0]["lr"])
    return lr


def _parse_step_from_checkpoint_dirname(*, name: str) -> int | None:
    """Parse step number from a directory name like 'checkpoint-step-123'.

    Args:
        name: Directory name.

    Returns:
        Parsed integer step, or None if it does not match.
    """
    m: re.Match[str] | None = re.fullmatch(pattern=r"checkpoint-step-(\d+)", string=name)
    if m is None:
        return None
    return int(m.group(1))


def _rotate_checkpoints(*, output_dir: Path, keep_last: int) -> None:
    """Keep only the most recent N checkpoint directories.

    Args:
        output_dir: Base output directory containing checkpoint subdirectories.
        keep_last: Number of most recent checkpoints to keep.
    """
    if int(keep_last) <= 0:
        return
    if not output_dir.exists():
        return

    ckpts: list[tuple[int, Path]] = []
    for p in output_dir.iterdir():
        if not p.is_dir():
            continue
        step: int | None = _parse_step_from_checkpoint_dirname(name=p.name)
        if step is None:
            continue
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
    """Save a PEFT adapter checkpoint and rotate older checkpoints.

    Args:
        model: Model to save.
        tokenizer: Tokenizer to save.
        output_dir: Base output directory.
        step: Global step number.
        keep_last: Number of checkpoints to keep.

    Raises:
        RuntimeError: If model is not a PEFT model (to avoid huge/full-model saves).
    """
    if not isinstance(model, PeftModel):
        raise RuntimeError(
            "Refusing to checkpoint a non-PEFT model. "
            "Pass an adapter_path or let the script create a LoRA adapter."
        )

    ckpt_dir: Path = output_dir / f"checkpoint-step-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_directory=str(ckpt_dir))
    tokenizer.save_pretrained(save_directory=str(ckpt_dir))
    _rotate_checkpoints(output_dir=output_dir, keep_last=keep_last)


def _decode_generated_only(
    *,
    tokenizer: Any,
    sequences: torch.Tensor,
    prompt_len: int,
) -> list[str]:
    """Decode only the generated continuation (excluding the prompt tokens).

    Args:
        tokenizer: HF tokenizer.
        sequences: Token ids, shape [B, T].
        prompt_len: Number of prompt tokens.

    Returns:
        List of decoded generated continuations (one per sequence).
    """
    cont_ids: list[torch.Tensor] = []
    bsz: int = int(sequences.shape[0])
    for b in range(bsz):
        seq: torch.Tensor = sequences[b]
        if int(prompt_len) >= int(seq.shape[0]):
            cont_ids.append(seq.new_zeros((0,), dtype=seq.dtype))
        else:
            cont_ids.append(seq[int(prompt_len) :])

    texts: list[str] = tokenizer.batch_decode(
        cont_ids,
        skip_special_tokens=True,
    )
    return texts


def _push_and_trim(*, buf: list[float], val: float, max_len: int) -> None:
    """Append to a rolling buffer and trim to max length.

    Args:
        buf: Buffer to update.
        val: Value to append.
        max_len: Maximum buffer length.
    """
    buf.append(float(val))
    if len(buf) > int(max_len):
        del buf[0]


def _mean(*, xs: list[float]) -> float:
    """Compute mean of a non-empty list.

    Args:
        xs: List of floats.

    Returns:
        Mean value.
    """
    if len(xs) == 0:
        return 0.0
    return float(sum(xs) / float(len(xs)))


def _get_sampling_temperature(*, cfg: GRPO_CONFIG) -> float:
    """Resolve a valid sampling temperature for GRPO.

    GRPO requires stochastic sampling to produce diverse group samples. If the
    config sets temperature=None, this function falls back to a safe default.

    Args:
        cfg: GRPO configuration.

    Returns:
        A temperature > 0.
    """
    if cfg.temperature is None:
        return 0.8
    t: float = float(cfg.temperature)
    if t <= 0.0:
        return 0.8
    return t


def main(*, adapter_path: Path | None = None) -> None:
    """Run group-normalized policy-gradient training on GSM8K.

    All hyperparameters and paths are sourced from the GRPO_CONFIG dataclass.

    Args:
        adapter_path: Optional PEFT adapter directory. If None, a fresh LoRA is created.
    """
    cfg: GRPO_CONFIG = GRPO_CONFIG()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model: torch.nn.Module
    tokenizer: Any
    model, tokenizer = _load_model_and_tokenizer(cfg=cfg, adapter_path=adapter_path)
    model.train()

    trainable_n: int
    total_n: int
    trainable_n, total_n = _count_trainable_params(model=model)
    print(
        f"Trainable params: {trainable_n} / {total_n} "
        f"({100 * trainable_n / total_n:.2f}%)"
    )

    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        params=[p for p in model.parameters() if bool(p.requires_grad)],
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

    step: int = 0
    optimizer.zero_grad(set_to_none=True)

    device: torch.device = next(model.parameters()).device

    logging_interval: int = int(cfg.loogging_interval)
    checkpoint_interval: int = int(cfg.checkpoint_interval)
    keep_last: int = int(cfg.keep_last_checkpoints)

    group_size: int = int(cfg.group_size)
    max_new_tokens: int = int(cfg.max_new_tokens)
    temperature: float = _get_sampling_temperature(cfg=cfg)
    top_p: float | None = cfg.top_p

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
                0.0,
                device=device,
                dtype=torch.float32,
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

                prompt_input_ids: torch.Tensor = prompt_inputs["input_ids"].to(device=device)
                prompt_attention_mask: torch.Tensor = prompt_inputs["attention_mask"].to(
                    device=device
                )

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

                with torch.no_grad():
                    gen_ids: torch.Tensor = model.generate(**gen_kwargs)

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

                prompt_len: int = int(prompt_input_ids.shape[1])
                prompt_lens: torch.Tensor = torch.full(
                    (int(group_size),),
                    fill_value=prompt_len,
                    device=group_input_ids.device,
                    dtype=torch.long,
                )

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
                advantages: torch.Tensor = (rewards_tensor - mean_reward) / (std_reward + 1e-8)

                logp_sums: torch.Tensor = _gather_logprobs_of_generated_tokens(
                    model=model,
                    input_ids=group_input_ids,
                    attention_mask=group_attention_mask,
                    prompt_lens=prompt_lens,
                ).to(dtype=torch.float32)

                adv_detached: torch.Tensor = advantages.detach()
                loss_q: torch.Tensor = -(adv_detached * logp_sums).mean()
                total_loss = total_loss + loss_q

            total_loss = total_loss / float(max(len(batch_questions), 1))
            total_loss = total_loss / float(grad_accum_steps)
            total_loss.backward()

            step += 1

            if step % int(grad_accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters=[p for p in model.parameters() if bool(p.requires_grad)],
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
                reward_std: float = float(_safe_std(x=rewards_step).detach().cpu().item())
                accuracy: float = reward_mean
            else:
                reward_mean = 0.0
                reward_std = 0.0
                accuracy = 0.0

            _push_and_trim(buf=loss_buf, val=loss_val, max_len=logging_interval)
            _push_and_trim(buf=r_mean_buf, val=reward_mean, max_len=logging_interval)
            _push_and_trim(buf=r_std_buf, val=reward_std, max_len=logging_interval)
            _push_and_trim(buf=acc_buf, val=accuracy, max_len=logging_interval)

            avg_loss: float = _mean(xs=loss_buf)
            avg_r_mean: float = _mean(xs=r_mean_buf)
            avg_r_std: float = _mean(xs=r_std_buf)
            avg_acc: float = _mean(xs=acc_buf)

            lr: float = _get_learning_rate(optimizer=optimizer)

            if logging_interval > 0 and step % logging_interval == 0:
                window: int = int(len(loss_buf))
                pbar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "r_mean": f"{avg_r_mean:.3f}",
                        "acc": f"{avg_acc:.3f}",
                        "r_std": f"{avg_r_std:.3f}",
                        "lr": f"{lr:.2e}",
                        "win": f"{window:d}",
                    }
                )
                print()
                print(
                    " | ".join(
                        [
                            f"[epoch={epoch + 1:0{len(str(cfg.epochs))}d} "
                            f"step={step:0{len(str(abs(len(loader))))}d}]",
                            f"loss(avg@{window})={avg_loss:.4f}",
                            f"reward_mean(avg@{window})={avg_r_mean:.3f}",
                            f"acc(avg@{window})={avg_acc:.3f}",
                            f"reward_std(avg@{window})={avg_r_std:.3f}",
                            f"lr={lr:.2e}",
                        ]
                    )
                )

            if checkpoint_interval > 0 and step % checkpoint_interval == 0:
                _save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=out_dir,
                    step=step,
                    keep_last=keep_last,
                )

        epoch_loss_mean: float = epoch_loss_sum / float(max(epoch_loss_count, 1))
        print(f"[epoch={epoch + 1}] epoch_loss_mean={epoch_loss_mean:.4f}")

    if not isinstance(model, PeftModel):
        raise RuntimeError(
            "Training finished but model is not a PEFT model. "
            "This script expects to train/save LoRA adapters only."
        )

    model.save_pretrained(save_directory=str(out_dir))
    tokenizer.save_pretrained(save_directory=str(out_dir))
    print("GRPO Training finished")


def _parse_optional_checkpoint_arg(*, argv: list[str]) -> Path | None:
    """Parse optional checkpoint path from argv[1].

    Args:
        argv: Full sys.argv.

    Returns:
        Path if provided, else None.
    """
    if len(argv) < 2:
        return None
    raw: str = argv[1].strip()
    if raw == "":
        return None
    return Path(raw)


if __name__ == "__main__":
    path: Path | None = _parse_optional_checkpoint_arg(argv=sys.argv)

    default_best: Path = Path("weights/final_rlm_lora/checkpoint-step-200")
    if path is None and default_best.exists():
        path = default_best

    main(adapter_path=path)
