"""
===============================================================================
BASE COGNITIVE AGENT CLASS (LLM GATEWAY INTEGRATED)
===============================================================================
"""

from __future__ import annotations
import json
import logging
from app.core.cognitive_config import cognitive_settings
from app.core.llm_inference_gateway import llm_gateway

logger = logging.getLogger("superlearn.agents.base")


class BaseAgent:
    """
    Standard base class representing a single-agent cognitive module.
    Delegates to LLM Gateway to seamlessly support Azure OpenAI and Ollama.
    """

    def __init__(self, model_name: str, temperature: float = 0.7) -> None:
        self.model_name = model_name
        self.temperature = temperature
        logger.info(f"Initializing BaseAgent: model={model_name}, temp={temperature}")

    async def invoke_chat(self, prompt: str, system_instruction: str = "") -> str:
        """
        Invokes LLM via the centralized inference gateway.
        """
        try:
            return await llm_gateway.generate_response(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=self.temperature,
            )
        except Exception as exc:
            logger.error(f"Agent invocation failed using gateway ({cognitive_settings.llm_provider}): {exc}")
            raise RuntimeError(
                f"Agent inference failed on provider '{cognitive_settings.llm_provider}'. Error: {exc}"
            ) from exc

    def parse_json(self, raw_text: str) -> dict:
        """
        Utility to clean markdown json code-blocks and parse dictionary output.
        """
        clean_text = raw_text.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json", 1)[1]
            if "```" in clean_text:
                clean_text = clean_text.split("```", 1)[0]
        elif "```" in clean_text:
            clean_text = clean_text.split("```", 1)[1]
            if "```" in clean_text:
                clean_text = clean_text.split("```", 1)[0]

        clean_text = clean_text.strip()
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            clean_text = clean_text[start_idx : end_idx + 1]

        try:
            return json.loads(clean_text)
        except Exception as exc:
            logger.error(f"JSON parsing error: {exc}. Raw text was:\n{raw_text}")
            raise ValueError(f"Failed to parse JSON response from LLM: {exc}")
