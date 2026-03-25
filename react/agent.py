from __future__ import annotations

from typing import Any

import torch
from langchain_core.tools import BaseTool

from tool_use.langchain.tool_inference import run_react_inference


class ReActAgent:
    """ReAct agent that combines reasoning (Phase 1), tools (Phase 2), and RAG (Phase 3).

    The agent follows a Thought-Action-Observation loop:
      1. Thought: model reasons inside <think>...</think>
      2. Action: model calls tools via @tool_name(args)->ID
      3. Observation: system executes tools, returns <tools>...</tools>
      4. Repeat until the model produces <answer>...</answer>

    This uses the dedicated ``run_react_inference`` loop which adds stale
    detection, forced finalization, message compression, and error escalation
    on top of the basic tool-use cycle.

    Args:
        model: Loaded HuggingFace causal LM (base + LoRA).
        tokenizer: Corresponding HuggingFace tokenizer.
        cfg: REACT_INFERENCE_CONFIG with system_prompt and generation params.
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
              - trace: List of step dicts with {"step", "type", "content"}.
              - details: Metadata about the run.
        """
        trace: list[dict[str, Any]] = run_react_inference(
            question=question,
            model=self._model,
            tokenizer=self._tokenizer,
            cfg=self._cfg,
            tools=self._tools,
        )

        # Extract final answer from the last trace entry
        final_answer: str = "<answer>null</answer>"
        for entry in reversed(trace):
            if entry.get("type") == "final_answer":
                final_answer = str(entry["content"])
                break

        return {
            "response": final_answer,
            "trace": trace,
            "details": {
                "stage": "react_agent",
                "num_steps": len(trace),
            },
        }
