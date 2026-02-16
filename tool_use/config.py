from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rlm.config import SYSTEM_PROMPT, SYSTEM_PROMPT_END
from utils.paths import find_parent_with_markers

MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME: str = "gsm8k"
DATASET_CONFIG: str = "main"

REPO_DIR: Path = find_parent_with_markers(start=Path.cwd())


# TOOL_SYSTEM_PROMPT: str = (
#     "TOOL USAGE RULES:\n"
#     "Tools may be invoked ONLY inside the <think>...</think> section.\n"
#     "Tools are NEVER allowed inside the <answer> section.\n"
#     "\n"
#     "TOOL CALL FORMAT:\n"
#     "A tool call must use the tag:\n"
#     "<tool id=INTEGER name=STRING args=DICT>\n"
#     "\n"
#     "Where:\n"
#     "- id: integer identifier for the tool call.\n"
#     "- name: exact tool name.\n"
#     "- args: dictionary of argument values.\n"
#     "\n"
#     "ID UNIQUENESS CONSTRAINT:\n"
#     "- Every tool id MUST be globally unique across the entire tool-usage loop.\n"
#     "- Reusing an id is strictly forbidden, even across different <think> blocks\n"
#     "  or different reasoning iterations.\n"
#     "\n"
#     "ID REFERENCES:\n"
#     "- The tag <id=INTEGER> may appear:\n"
#     "  • Inside tool arguments → replaced by the referenced tool output "
#     "before executing the current tool.\n"
#     "  • Inside <answer> → replaced by the referenced tool output "
#     "to form the final validated answer.\n"
#     "- Referencing a tool that has not executed yet is forbidden.\n"
#     "\n"
#     "ITERATION LOOP:\n"
#     "- The model may perform up to 3 reasoning iterations when tools are used.\n"
#     "- Each iteration follows this structure:\n"
#     "    <think>...optional tool calls...</think>\n"
#     "    (system executes tools and returns a <tools> block)\n"
#     "\n"
#     "- The loop stops when:\n"
#     "  • An <answer> section is produced, OR\n"
#     "  • 3 iterations are reached.\n"
#     "\n"
#     "FAILURE CONDITIONS:\n"
#     "- If an iteration contains neither a tool call nor an <answer>, "
#     "the validated result becomes <answer></answer>.\n"
#     "- If 3 iterations occur without an <answer>, the validated result "
#     "becomes <answer></answer>.\n"
#     "\n"
#     "THE <tools> BLOCK FORMAT:\n"
#     "- After a <think> containing tool calls, the system returns:\n"
#     "  <tools>\n"
#     "  {\n"
#     "      TOOL_ID: {\n"
#     "          tool: STRING,\n"
#     "          args: DICT,\n"
#     "          output: STRING\n"
#     "      },\n"
#     "      ...\n"
#     "  }\n"
#     "  </tools>\n"
#     "\n"
#     "- TOOL_ID exactly matches the id used in <tool id=...>.\n"
#     "- 'output' is the exact textual result returned by the tool.\n"
#     "- The model MUST read this block before producing the next <think>.\n"
#     "- Tool outputs are immutable and must not be altered by the model.\n"
#     "\n"
#     "VALID RESPONSE PATTERNS:\n"
#     "- Without tools:\n"
#     "    <question>...</question><think>...</think><answer>...</answer>\n"
#     "\n"
#     "- With tools:\n"
#     "    <question>...</question>\n"
#     "    <think>...<tool ...>...</think>\n"
#     "    <tools>...</tools>\n"
#     "    ... reps of <think> + <tools>\n"
#     "    <think>...</think>\n"
#     "    <answer>...</answer>\n"
#     "\n"
#     "---\n"
#     "FIRST EXAMPLE:\n"
#     "\n"
#     "Input:\n"
#     "<question>what is the result of sqrt(45234)?</question>\n"
#     "\n"
#     "Response:\n"
#     "<think>sqrt means square root. I compute it using exponent 1/2. "
#     "The result is <tool id=1 name=calculator args={expression='45234**(1/2)'}>.</think>\n"
#     "<answer><id=1></answer>\n"
#     "\n"
#     "Parsed answer:\n"
#     "<answer>212.6828624971932</answer>\n"
#     "\n"
#     "---\n"
#     "SECOND EXAMPLE:\n"
#     "\n"
#     "Input:\n"
#     "<question>what is the result of sqrt(45234) + (5.23 * 4.83)?</question>\n"
#     "\n"
#     "Response:\n"
#     "<think>I compute each term separately. "
#     "sqrt term: <tool id=1 name=calculator args={expression='45234**(1/2)'}>. "
#     "Product term: <tool id=2 name=calculator args={expression='5.23 * 4.83'}>. "
#     "Sum: <tool id=3 name=calculator args={expression='<id=1> + <id=2>'}>.</think>\n"
#     "<answer><id=3></answer>\n"
#     "\n"
#     "Parsed answer:\n"
#     "<answer>237.9437624971932</answer>\n"
#     "\n"
#     "---\n"
#     "THIRD EXAMPLE (MULTI-STEP TOOL USE):\n"
#     "\n"
#     "Input:\n"
#     "<question>what is the result of sqrt(45234) truncated to 2 decimals?</question>\n"
#     "\n"
#     "First response:\n"
#     "<think>I compute the square root: "
#     "<tool id=1 name=calculator args={expression='45234**(1/2)'}>.</think>\n"
#     "\n"
#     "System tool execution:\n"
#     "<tools>\n"
#     "{\n"
#     "    1: {\n"
#     "        tool: 'calculator',\n"
#     "        args: {'expression': '45234**(1/2)'},\n"
#     "        output: '212.6828624971932'\n"
#     "    }\n"
#     "}\n"
#     "</tools>\n"
#     "\n"
#     "Second response:\n"
#     "<think>Truncating 212.6828624971932 to 2 decimals gives 212.68.</think>\n"
#     "<answer>212.68</answer>\n"
#     "\n"
#     "Parsed answer:\n"
#     "<answer>212.68</answer>\n"
# )


TOOL_SYSTEM_PROMPT: str = (
    "TOOL USAGE RULES:\n"
    "Tools may be invoked ONLY inside the <think>...</think> section.\n"
    "Tools are NEVER allowed inside the <answer> section.\n"
    "\n"
    "TOOL CALL FORMAT:\n"
    "A tool call must use the function-call style:\n"
    '@TOOL_NAME(arg1="...", arg2="...", ...):i\n'
    "\n"
    "Where:\n"
    "- TOOL_NAME: exact tool name.\n"
    "- argN: named arguments passed to the tool.\n"
    "- i: integer identifier (id) that labels the output of this tool call.\n"
    "\n"
    "ID UNIQUENESS CONSTRAINT:\n"
    "- Every tool id MUST be globally unique across the entire tool-usage loop.\n"
    "- Reusing an id is strictly forbidden, even across different <think> blocks\n"
    "  or different reasoning iterations.\n"
    "\n"
    "ID REFERENCES:\n"
    "- The reference syntax is @i (where i is an integer tool id).\n"
    "- @i may appear:\n"
    "  • Inside tool arguments → replaced by the referenced tool output "
    "before executing the current tool.\n"
    "  • Inside <answer> → replaced by the referenced tool output "
    "to form the final validated answer.\n"
    "- Referencing a tool that has not executed yet is forbidden.\n"
    "\n"
    "ITERATION LOOP:\n"
    "- The model may perform up to 3 reasoning iterations when tools are used.\n"
    "- Each iteration follows this structure:\n"
    "    <think>...optional tool calls...</think>\n"
    "    (system executes tools and returns a <tools> block)\n"
    "\n"
    "- The loop stops when:\n"
    "  • An <answer> section is produced, OR\n"
    "  • 3 iterations are reached.\n"
    "\n"
    "FAILURE CONDITIONS:\n"
    "- If an iteration contains neither a tool call nor an <answer>, "
    "the validated result becomes <answer></answer>.\n"
    "- If 3 iterations occur without an <answer>, the validated result "
    "becomes <answer></answer>.\n"
    "\n"
    "THE <tools> BLOCK FORMAT:\n"
    "- After a <think> containing tool calls, the system returns:\n"
    "  <tools>\n"
    "  {\n"
    "      TOOL_ID: {\n"
    "          tool: STRING,\n"
    "          args: DICT,\n"
    "          output: STRING\n"
    "      },\n"
    "      ...\n"
    "  }\n"
    "  </tools>\n"
    "\n"
    "- TOOL_ID exactly matches the id used after ':' in a tool call (...):i.\n"
    "- 'output' is the exact textual result returned by the tool.\n"
    "- The model MUST read this block before producing the next <think>.\n"
    "- Tool outputs are immutable and must not be altered by the model.\n"
    "\n"
    "VALID RESPONSE PATTERNS:\n"
    "- Without tools:\n"
    "    <question>...</question><think>...</think><answer>...</answer>\n"
    "\n"
    "- With tools:\n"
    "    <question>...</question>\n"
    "    <think>...@tool_name(...):i ...</think>\n"
    "    <tools>...</tools>\n"
    "    ... reps of <think> + <tools>\n"
    "    <think>...</think>\n"
    "    <answer>...</answer>\n"
    "\n"
    "---\n"
    "FIRST EXAMPLE:\n"
    "\n"
    "Input:\n"
    "<question>what is the result of sqrt(45234)?</question>\n"
    "\n"
    "Response:\n"
    "<think>sqrt means square root. I compute it using exponent 1/2. "
    'The result is @calculator(expression="45234**(1/2)"):1.</think>\n'
    "<answer>@1</answer>\n"
    "\n"
    "Parsed answer:\n"
    "<answer>212.6828624971932</answer>\n"
    "\n"
    "---\n"
    "SECOND EXAMPLE:\n"
    "\n"
    "Input:\n"
    "<question>what is the result of sqrt(45234) + (5.23 * 4.83)?</question>\n"
    "\n"
    "Response:\n"
    "<think>I compute each term separately. "
    'sqrt term: @calculator(expression="45234**(1/2)"):1. '
    'Product term: @calculator(expression="5.23 * 4.83"):2. '
    'Sum: @calculator(expression="@1 + @2"):3.</think>\n'
    "<answer>@3</answer>\n"
    "\n"
    "Parsed answer:\n"
    "<answer>237.9437624971932</answer>\n"
    "\n"
    "---\n"
    "THIRD EXAMPLE (MULTI-STEP TOOL USE):\n"
    "\n"
    "Input:\n"
    "<question>what is the result of sqrt(45234) truncated to 2 decimals?</question>\n"
    "\n"
    "First response:\n"
    "<think>I compute the square root: "
    '@calculator(expression="45234**(1/2)"):1.</think>\n'
    "\n"
    "System tool execution:\n"
    "<tools>\n"
    "{\n"
    "    1: {\n"
    "        tool: 'calculator',\n"
    "        args: {'expression': '45234**(1/2)'},\n"
    "        output: '212.6828624971932'\n"
    "    }\n"
    "}\n"
    "</tools>\n"
    "\n"
    "Second response:\n"
    "<think>Truncating 212.6828624971932 to 2 decimals gives 212.68.</think>\n"
    "<answer>212.68</answer>\n"
    "\n"
    "Parsed answer:\n"
    "<answer>212.68</answer>\n"
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
