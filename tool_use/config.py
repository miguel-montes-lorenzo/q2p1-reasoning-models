from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rlm.config import SYSTEM_PROMPT, SYSTEM_PROMPT_END
from utils import find_parent_with_markers

MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME: str = "gsm8k"
DATASET_CONFIG: str = "main"

REPO_DIR: Path = find_parent_with_markers(start=Path.cwd())


TOOL_SYSTEM_PROMPT: str = (
    "TOOLS:\n"
    "The <tool ...> tag is a special tag that can be used only inside the <think> tag "
    "(this is an exception to the non-nesting tag rule, but the nesting depth is "
    "restricted to 1 inside the think tag). All calls to tools will have 3 arguments:\n"
    "   - id: identifier used to identify the tool (must be an integer) (cannot be "
    "repeated across tools inside the same <think> tag token)\n"
    "   - name: name of the tool to use\n"
    "   - args: dictionary with the argument values for the tool\n"
    "\n"
    "The <id=...> tag is a special tag that can be used inside the <tool ...> tag "
    "argument values, or inside the <answer> tag. Appearances of the <id=...> tag "
    "inside <tool ...> arguments will be formatted with the output value of the "
    "identified tool before this tool evaluation (make sure to avoid cross-references). "
    "Appearances of the <id=...> tag inside <answer> will be formatted for the answer "
    "evaluation.\n"
    "\n"
    "The model will have up to 3 iterations of responses to use tools. The tool loop "
    "will end either after the 3rd iteration or when an iteration contains the "
    "<answer> tag. If the model reaches the 3rd iteration without an answer, a blank "
    "answer <answer></answer> will be validated. If an iteration does not contain "
    "either <tool ...> tags or an <answer> tag, a blank answer <answer></answer> will "
    "be validated.\n"
    "\n"
    "---\n"
    "First Example:\n"
    "\n"
    "- First input:\n"
    "<question>what is the result of sqrt(45234)?</question>\n"
    "\n"
    "- First response:\n"
    "<think>sqrt means square root. The calculator tool does not have a square root "
    "operator, but I can use a fractional exponent. The result of sqrt(45234) is "
    "<tool id=1 name=calculator args={expression='45234**(1/2)'}></think>\n"
    "<answer><id=1></answer>\n"
    "\n"
    "\n"
    "---\n"
    "Second Example:\n"
    "\n"
    "- First input:\n"
    "<question>what is the result of sqrt(45234) + (5.23 * 4.83)?</question>\n"
    "\n"
    "- First response:\n"
    "<think>sqrt means square root. The calculator tool does not have a square root "
    "operator, but I can use a fractional exponent. The result of sqrt(45234) is "
    "<tool id=1 name=calculator args={expression='45234**(1/2)'}>. (5.23 * 4.83) is "
    "not a trivial multiplication, so it would be safer to use the calculator tool. "
    "The result of (5.23 * 4.83) is <tool id=2 name=calculator args={expression='5.23 "
    "* 4.83'}>. In order to get the result of sqrt(45234) + (5.23 * 4.83), I need to "
    "sum the results of both terms of the expression. As the results are being "
    "computed in this same think block with the calculator tool, I can reference them "
    "using their corresponding id tags. The result of sqrt(45234) + (5.23 * 4.83) is "
    "<tool id=3 name=calculator args={expression='<id=1> + <id=2>'}></think>\n"
    "<answer><id=3></answer>\n"
    "\n"
    "\n"
    "---\n"
    "Third Example (if the tools functionality does not allow answering in one step, "
    "do it in more steps, up to 3):\n"
    "\n"
    "- First input:\n"
    "<question>what is the result of sqrt(45234) truncated to 2 decimals?</question>\n"
    "\n"
    "- First response:\n"
    "<think>sqrt means square root. The calculator tool does not have a square root "
    "operator, but I can use a fractional exponent. The result of sqrt(45234) is "
    "<tool id=1 name=calculator args={expression='45234**(1/2)'}></think>\n"
    "\n"
    "- Second input:\n"
    "<question>what is the result of sqrt(45234) truncated to 2 decimals</question>\n"
    "<think>sqrt means square root. The calculator tool does not have a square root "
    "operator, but I can use a fractional exponent. The result is <tool id=1 "
    "name=calculator args={expression='45234**(1/2)'}></think>\n"
    "<tools>\n"
    "{\n"
    "    1: {\n"
    "        tool: 'calculator',\n"
    "        args: {\n"
    "            'expression': '45234**(1/2)'\n"
    "        }\n"
    "        output: '212.6828624971932'\n"
    "    }\n"
    "}\n"
    "</tools>\n"
    "\n"
    "- Second response:\n"
    "<think>The result of sqrt(45234) is 212.6828624971932. Truncating "
    "212.6828624971932 to 0 decimals results in 212.</think>\n"
    "<answer>212</answer>\n"
)


USE_4_BIT: bool = True
MAX_SEQ_LEN: int = 1024

MAX_NEW_TOKENS: int = 1024
TEMPERATURE: float = 0.8
TOP_P: float = 0.95


@dataclass(frozen=True)
class SFT_CONFIG:
    """Configuration container for Supervised Fine-Tuning (SFT).

    This class centralizes all parameters required for SFT training,
    checkpointing, and evaluation/check scripts. It is intended to be
    instantiated once and passed to training and inference utilities
    to ensure consistent behavior across scripts.

    Attributes:
        model_name: Hugging Face model identifier of the base causal LM.
        dataset_name: Hugging Face dataset name used for SFT.
        dataset_config: Dataset configuration or subset name passed to
            `load_dataset(..., name=dataset_config)`.

        use_4bit: Whether to load the base model using 4-bit NF4 quantization.
        max_seq_len: Maximum total sequence length (prompt + completion).

        system_prompt: System prompt prepended when formatting inputs using
            a chat template.

        max_new_tokens: Maximum number of tokens generated during inference
            in check/evaluation scripts.
        temperature: Sampling temperature for generation. If None, generation
            defaults to deterministic decoding.
        top_p: Nucleus sampling probability mass. Only used when sampling
            is enabled.

        epochs: Number of training epochs for SFT.
        lr: Learning rate used during SFT optimization.
        batch_size_questions: Per-device batch size measured in questions
            (training examples).

        loogging_interval: Number of steps between metric logging.
        checkpoint_directory: Directory where LoRA adapters and tokenizer
            checkpoints are saved.
        checkpoint_interval: Number of steps between checkpoint saves.
        keep_last_checkpoints: Maximum number of recent checkpoints to retain.
    """

    model_name: str = MODEL_NAME
    dataset_name: str = DATASET_NAME
    dataset_config: str = DATASET_CONFIG

    use_4bit: bool = USE_4_BIT
    max_seq_len: int = MAX_SEQ_LEN

    system_prompt: str = f"{SYSTEM_PROMPT}{TOOL_SYSTEM_PROMPT}{SYSTEM_PROMPT_END}"

    # Generation hyperparameters (check script)
    do_sample: bool = False
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float | None = TEMPERATURE if do_sample else None
    top_p: float | None = TOP_P if do_sample else None

    # Training hyperparameters (SFT)
    epochs: int = 1
    lr: float = 2e-4
    batch_size_questions: int = 4

    # tool use
    max_calls: int = 3

    # Training Management
    loogging_interval: int = 10
    checkpoint_directory: Path = REPO_DIR / "weights/tool_use/sft_lora"
    checkpoint_interval: int = 200
    keep_last_checkpoints: int = 2


@dataclass(frozen=True)
class INFERENCE_CONFIG:
    """Configuration container for Supervised Fine-Tuning (SFT).

    This class centralizes all parameters required for SFT training,
    checkpointing, and evaluation/check scripts. It is intended to be
    instantiated once and passed to training and inference utilities
    to ensure consistent behavior across scripts.

    Attributes:
        model_name: Hugging Face model identifier of the base causal LM.
        dataset_name: Hugging Face dataset name used for SFT.
        dataset_config: Dataset configuration or subset name passed to
            `load_dataset(..., name=dataset_config)`.

        use_4bit: Whether to load the base model using 4-bit NF4 quantization.
        max_seq_len: Maximum total sequence length (prompt + completion).

        system_prompt: System prompt prepended when formatting inputs using
            a chat template.

        max_new_tokens: Maximum number of tokens generated during inference
            in check/evaluation scripts.
        temperature: Sampling temperature for generation. If None, generation
            defaults to deterministic decoding.
        top_p: Nucleus sampling probability mass. Only used when sampling
            is enabled.

        epochs: Number of training epochs for SFT.
        lr: Learning rate used during SFT optimization.
        batch_size_questions: Per-device batch size measured in questions
            (training examples).

        loogging_interval: Number of steps between metric logging.
        checkpoint_directory: Directory where LoRA adapters and tokenizer
            checkpoints are saved.
        checkpoint_interval: Number of steps between checkpoint saves.
        keep_last_checkpoints: Maximum number of recent checkpoints to retain.
    """

    model_name: str = MODEL_NAME
    dataset_name: str = DATASET_NAME
    dataset_config: str = DATASET_CONFIG

    use_4bit: bool = USE_4_BIT
    max_seq_len: int = MAX_SEQ_LEN

    system_prompt: str = f"{SYSTEM_PROMPT}{TOOL_SYSTEM_PROMPT}{SYSTEM_PROMPT_END}"

    # Generation hyperparameters (check script)
    do_sample: bool = True
    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float | None = TEMPERATURE if do_sample else None
    top_p: float | None = TOP_P if do_sample else None

    # tool use
    max_calls: int = 3

    checkpoint_directory: Path = REPO_DIR / "weights/tool_use/sft_lora/best_checkpoint"
