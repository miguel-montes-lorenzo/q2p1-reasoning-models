from __future__ import annotations

from typing import Any

import torch
from langchain_core.tools import BaseTool

from tool_use.langchain.tool_inference import run_tool_use_inference


class ReActAgent:
    """ReAct agent that combines reasoning (Phase 1), tools (Phase 2), and RAG (Phase 3).

    The agent follows a Thought-Action-Observation loop:
      1. Thought: model reasons inside <think>...</think>
      2. Action: model calls tools via @tool_name(args)->ID
      3. Observation: system executes tools, returns <tools>...</tools>
      4. Repeat until the model produces <answer>...</answer>

    This reuses the existing tool-use inference loop which already implements
    this exact cycle with the <think>/<tools>/<answer> XML tag protocol.

    Args:
        model: Loaded HuggingFace causal LM (base + LoRA).
        tokenizer: Corresponding HuggingFace tokenizer.
        cfg: Tool-use inference config with system_prompt and generation params.
        tools: List of LangChain tools (calculator, knowledge_base_search, etc.).
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        cfg: Any,
        tools: list[BaseTool],
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._cfg = cfg
        self._tools = tools

    def run(self, question: str) -> dict[str, Any]:
        """Run the ReAct loop for a single question.

        Args:
            question: The user's question.

        Returns:
            Dict with keys:
              - response: The final answer string.
              - trace: List of step dicts for the UI.
              - details: Metadata about the run.
        """
        full_output, parsed_answer, step_contents = run_tool_use_inference(
            question=question,
            model=self._model,
            tokenizer=self._tokenizer,
            cfg=self._cfg,
            tools=self._tools,
            formatted_references=True,
        )

        trace: list[dict[str, Any]] = [
            {"step": i, "content": content}
            for i, content in enumerate(step_contents)
        ]

        return {
            "response": parsed_answer,
            "trace": trace,
            "details": {"stage": "react_agent", "num_steps": len(step_contents)},
        }
