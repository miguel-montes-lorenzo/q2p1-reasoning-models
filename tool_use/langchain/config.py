# tool_use/langchain/config.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rlm.config import SYSTEM_PROMPT, SYSTEM_PROMPT_END
from utils.paths import find_parent_with_markers

MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
DATASET_NAME: str = "gsm8k"
DATASET_CONFIG: str = "main"

REPO_DIR: Path = find_parent_with_markers(start=Path.cwd())


MAX_THINK_CALLS: int = 4


# TOOL_SYSTEM_PROMPT: str = (
#     "TOOL USAGE RULES:\n"
#     "Tools may be invoked ONLY inside the <think>...</think> section.\n"
#     "Tools are NEVER allowed inside the <answer> section.\n"
#     "\n"
#     "TOOL CALL FORMAT:\n"
#     "A tool call must use the function-call style:\n"
#     '@TOOL_NAME(arg1="...", arg2="...", ...)->i\n'
#     "\n"
#     "Where:\n"
#     "- TOOL_NAME: exact tool name.\n"
#     "- argN: named arguments passed to the tool.\n"
#     "- i: integer identifier (id) that labels the output of this tool call.\n"
#     "\n"
#     "ID UNIQUENESS CONSTRAINT:\n"
#     "- Every tool id MUST be globally unique across the entire tool-usage loop.\n"
#     "- Reusing an id is strictly forbidden, even across different <think> blocks\n"
#     "  or different reasoning iterations.\n"
#     "\n"
#     "ID REFERENCES:\n"
#     "- The reference syntax is @i (where i is an integer tool id).\n"
#     "- @i may appear:\n"
#     "  • Inside tool arguments → replaced by the referenced tool output "
#     "before executing the current tool.\n"
#     "  • Inside <answer> → replaced by the referenced tool output "
#     "to form the final validated answer.\n"
#     "- Referencing a tool that has not executed yet is forbidden.\n"
#     "\n"
#     "WHEN TO USE @:\n"
#     "- Use of @ will ALLWAYS be interpreted as an attempt to call a function or "
#     "reference an id. Plain use of @ in your thinking process will be interpreted as a "
#     "wrongly formated call, and result in an error.\n"
#     "\n"
#     "ITERATION LOOP:\n"
#     f"- The model may perform up to {MAX_THINK_CALLS} <think> iterations "
#     "when tools are used.\n"
#     f"- Using fewer than {MAX_THINK_CALLS} iterations is preferred.\n"
#     f"- However, it is better to use more iterations (up to {MAX_THINK_CALLS}) "
#     "to make tool calls that improve reasoning quality than to give an incorrect answer.\n"
#     "- Each iteration follows this structure:\n"
#     "    <think>...optional tool calls...</think>\n"
#     "    (system executes tools and returns either a <tools> block or an <error> block)\n"
#     "\n"
#     "- The loop stops when:\n"
#     "  • An <answer> section is produced, OR\n"
#     f"  • {MAX_THINK_CALLS} iterations are reached.\n"
#     "\n"
#     "FAILURE CONDITIONS:\n"
#     "- If a tool call has a format or semantic error, the system returns:\n"
#     "    <error>ERROR_MESSAGE</error>\n"
#     "  and any <answer> generated in that same iteration MUST be ignored.\n"
#     "- If an <error> block appears, it indicates a severe formatting mistake in the\n"
#     "  immediately previous <think>...</think>. You MUST restructure the next <think>\n"
#     "  you generate to strictly follow the tool-call format and avoid repeating it.\n"
#     "- When an <error> block is returned, it plays the same structural role as <tools>,\n"
#     "  and the model must read it before the next <think>.\n"
#     f"- If the maximum number of iterations ({MAX_THINK_CALLS}) is reached without a valid "
#     "<answer>, the validated result becomes <answer>null</answer>.\n"
#     "\n"
#     "ALLOWED ERROR MESSAGES:\n"
#     "- Incorrect use of @, allowed formats are: "
#     "{@available_funcion_name(arg1=..., arg2=..., ...)->int_id, @defined_int_id}\n"
#     "- Call to unexistent function: @used_function_name\n"
#     "- Reference to undefined id: @used_int_id\n"
#     "- Use of unexistent arguments for function @used_function: used_argument\n"
#     "\n"
#     "THE <tools> BLOCK FORMAT:\n"
#     "- After a <think> containing valid tool calls, the system returns:\n"
#     "  <tools>\n"
#     "  {\n"
#     "      TOOL_ID: {\n"
#     "          tool: STRING,\n"
#     "          successful_execution: BOOL,\n"
#     "          args: DICT,\n"
#     "          output: STRING\n"
#     "      },\n"
#     "      ...\n"
#     "  }\n"
#     "  </tools>\n"
#     "\n"
#     "- TOOL_ID exactly matches the id used after '->' in a tool call (...)->i.\n"
#     "- 'successful_execution' is True iff the tool finished without error, else False.\n"
#     "- 'output' is the exact textual result returned by the tool (or the error text).\n"
#     "- Tool outputs are immutable and must not be altered by the model.\n"
#     "\n"
#     "VALID RESPONSE PATTERNS:\n"
#     "- Without tools:\n"
#     "    <question>...</question><think>...</think><answer>...</answer>\n"
#     "\n"
#     "- With tools:\n"
#     "    <question>...</question>\n"
#     "    <think>...@tool_name(...)->i ...</think>\n"
#     "    <tools>...</tools> OR <error>...</error>\n"
#     "    ... reps of <think> + (<tools> OR <error>)\n"
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
#     'The result is @calculator(expression="45234**(1/2)")->1.</think>\n'
#     "<answer>@1</answer>\n"
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
#     'sqrt term: @calculator(expression="45234**(1/2)")->1. '
#     'Product term: @calculator(expression="5.23 * 4.83")->2. '
#     'Sum: @calculator(expression="@1 + @2")->3.</think>\n'
#     "<answer>@3</answer>\n"
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
#     '@calculator(expression="45234**(1/2)")->1.</think>\n'
#     "\n"
#     "System tool execution:\n"
#     "<tools>\n"
#     "{\n"
#     "    1: {\n"
#     "        tool: 'calculator',\n"
#     "        successful_execution: True,\n"
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
    '@TOOL_NAME(arg1="...", arg2="...", ...)->ID\n'
    "\n"
    "Where:\n"
    "- TOOL_NAME: exact tool name.\n"
    "- argN: named arguments passed to the tool.\n"
    "- ID: identifier (id) that labels the output of this tool call.\n"
    "- IMPORTANT: ID MUST be a concrete alphanumeric literal (letters/digits/underscore),\n"
    "  e.g. ->1, ->2, ->15, ->i1, ->step_2. Never write placeholders like ->ID.\n"
    "\n"
    "TOOL ARGUMENTS (STRICT):\n"
    "- Before calling any tool, read its TOOL DESCRIPTION below.\n"
    "- You MUST use the exact argument names from the tool description/signature.\n"
    "- Do NOT invent arguments (e.g., expression1, query2, top_k, etc.) unless they\n"
    "  are explicitly listed for that tool.\n"
    "- Argument types and formatting MUST follow the tool description (e.g. strings,\n"
    "  numbers, required vs optional args, valid ranges).\n"
    "- If you are unsure about an argument name/type, do not call the tool; instead,\n"
    "  revise your <think> and either call a tool you understand or answer without tools.\n"
    "\n"
    "ID UNIQUENESS CONSTRAINT:\n"
    "- Every tool id MUST be globally unique across the entire tool-usage loop.\n"
    "- Reusing an id is strictly forbidden, even across different <think> blocks\n"
    "  or different reasoning iterations.\n"
    "\n"
    "ID REFERENCES:\n"
    "- The reference syntax is @ID (where ID is an alphanumeric id: letters/digits/_).\n"
    "- @ID may appear:\n"
    "  • Inside tool arguments → replaced by the referenced tool output "
    "before executing the current tool.\n"
    "  • Inside <answer> → replaced by the referenced tool output "
    "to form the final validated answer.\n"
    "- Referencing a tool that has not executed yet is forbidden.\n"
    "\n"
    "WHEN TO USE @:\n"
    "- Use of @ will ALLWAYS be interpreted as an attempt to call a function or "
    "reference an id. Plain use of @ in your thinking process will be interpreted as a "
    "wrongly formated call, and result in an error.\n"
    "\n"
    "ITERATION LOOP:\n"
    f"- The model may perform up to {MAX_THINK_CALLS} <think> iterations "
    "when tools are used.\n"
    f"- Using fewer than {MAX_THINK_CALLS} iterations is preferred.\n"
    f"- However, it is better to use more iterations (up to {MAX_THINK_CALLS}) "
    "to make tool calls that improve reasoning quality than to give an incorrect answer.\n"
    "- Each iteration follows this structure:\n"
    "    <think>...optional tool calls...</think>\n"
    "    (system executes tools and returns either a <tools> block or an <error> block)\n"
    "\n"
    "- The loop stops when:\n"
    "  • An <answer> section is produced, OR\n"
    f"  • {MAX_THINK_CALLS} iterations are reached.\n"
    "\n"
    "FAILURE CONDITIONS:\n"
    "- If a tool call has a format or semantic error, the system returns:\n"
    "    <error>ERROR_MESSAGE</error>\n"
    "  and any <answer> generated in that same iteration MUST be ignored.\n"
    "- If an <error> block appears, it indicates a severe formatting mistake in the\n"
    "  immediately previous <think>...</think>. You MUST restructure the next <think>\n"
    "  you generate to strictly follow the tool-call format and avoid repeating it.\n"
    "- When an <error> block is returned, it plays the same structural role as <tools>,\n"
    "  and the model must read it before the next <think>.\n"
    f"- If the maximum number of iterations ({MAX_THINK_CALLS}) is reached without a valid "
    "<answer>, the validated result becomes <answer>null</answer>.\n"
    "\n"
    "ALLOWED ERROR MESSAGES:\n"
    "- Incorrect use of @, allowed formats are: "
    "{@available_funcion_name(arg1=..., arg2=..., ...)->int_id, @defined_int_id}\n"
    "- Call to unexistent function: @used_function_name\n"
    "- Reference to undefined id: @used_int_id\n"
    "- Use of unexistent arguments for function @used_function: used_argument\n"
    "\n"
    "THE <tools> BLOCK FORMAT:\n"
    "- After a <think> containing valid tool calls, the system returns:\n"
    "  <tools>\n"
    "  {\n"
    "      TOOL_ID: {\n"
    "          tool: STRING,\n"
    "          successful_execution: BOOL,\n"
    "          args: DICT,\n"
    "          output: STRING\n"
    "      },\n"
    "      ...\n"
    "  }\n"
    "  </tools>\n"
    "\n"
    "- TOOL_ID exactly matches the id used after '->' in a tool call (...)->ID.\n"
    "- 'successful_execution' is True iff the tool finished without error, else False.\n"
    "- 'output' is the exact textual result returned by the tool (or the error text).\n"
    "- Tool outputs are immutable and must not be altered by the model.\n"
    "\n"
    "VALID RESPONSE PATTERNS:\n"
    "- Without tools:\n"
    "    <question>...</question><think>...</think><answer>...</answer>\n"
    "\n"
    "- With tools:\n"
    "    <question>...</question>\n"
    "    <think>...@tool_name(...)->ID ...</think>\n"
    "    <tools>...</tools> OR <error>...</error>\n"
    "    ... reps of <think> + (<tools> OR <error>)\n"
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
    'The result is @calculator(expression="45234**(1/2)")->1.</think>\n'
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
    'sqrt term: @calculator(expression="45234**(1/2)")->1. '
    'Product term: @calculator(expression="5.23 * 4.83")->2. '
    'Sum: @calculator(expression="@1 + @2")->3.</think>\n'
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
    '@calculator(expression="45234**(1/2)")->1.</think>\n'
    "\n"
    "System tool execution:\n"
    "<tools>\n"
    "{\n"
    "    1: {\n"
    "        tool: 'calculator',\n"
    "        successful_execution: True,\n"
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
    max_calls: int = MAX_THINK_CALLS

    checkpoint_directory: Path = REPO_DIR / "weights/tool_use/sft_lora/best_checkpoint"
