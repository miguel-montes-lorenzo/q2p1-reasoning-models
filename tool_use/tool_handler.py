import json
import re

from tool_use.tools import AVAILABLE_TOOLS_SCHEMA, tools_map


def parse_and_execute_tool_call(model_output: str):
    """
    Args:
        model_output (str): The raw text response from the LLM.

    Returns:
        dict: A dictionary containing 'executed': True/False and 'result': str.
    """

    # 1. Attempt to extract JSON from the output using Regex
    # We look for content inside ```json ... ``` or just { ... }
    json_str = None

    # Pattern to capture code blocks
    code_block_pattern = r"```json(.*?)```"
    matches = re.findall(
        pattern=code_block_pattern, string=model_output, flags=re.DOTALL
    )

    if matches:
        json_str = matches[0].strip()
    elif model_output.strip().startswith("{") and model_output.strip().endswith("}"):
        # Fallback if the model returns raw JSON without markdown
        json_str = model_output.strip()

    if not json_str:
        return {"executed": False, "result": model_output}

    # 2. Parse JSON and Execute Function
    try:
        tool_call = json.loads(s=json_str)

        tool_name = tool_call.get("tool")
        tool_args = tool_call.get("args", {})

        # Check if the tool exists in our map
        if tool_name in tools_map:
            print(
                f"--- [SYSTEM] Executing tool: {tool_name} with args: {tool_args} ---"
            )

            function_to_call = tools_map[tool_name]

            # Execute the function with unpacked arguments
            execution_result = function_to_call(**tool_args)

            return {"executed": True, "result": execution_result}
        else:
            return {
                "executed": True,
                "result": f"Error: Tool '{tool_name}' not defined.",
            }

    except json.JSONDecodeError:
        return {
            "executed": False,
            "result": "Error: Failed to decode JSON from model output.",
        }
    except Exception as e:
        return {"executed": True, "result": f"Error executing tool: {e}"}


# System Prompt construction
# This instructs the model to use the defined JSON format for tool calls.
SYSTEM_PROMPT_TOOLS: str = f"""
        You are a helpful assistant with access to external tools.

        AVAILABLE TOOLS (JSON Schema):
        {json.dumps(obj=AVAILABLE_TOOLS_SCHEMA, indent=2)}

        INSTRUCTIONS:
        1. If the user asks a question that requires a tool (math, facts, or books), YOU MUST RESPOND ONLY WITH A JSON BLOCK.
        2. The JSON format must be strictly:
        ```json
        {{
            "tool": "tool_name",
            "args": {{
                "argument_name": "value"
            }}
        }}
        3. If no tool is needed, respond with normal text.
        """
