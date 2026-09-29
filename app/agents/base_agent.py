"""
===============================================================================
BASE COGNITIVE AGENT CLASS (LANGCHAIN AGENTIC RAG & INFERENCE GATEWAY INTEGRATED)
===============================================================================
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from app.core.cognitive_config import cognitive_settings
from app.core.llm_inference_gateway import llm_gateway
from app.agents.rag_tools import get_agent_tools

logger = logging.getLogger("superlearn.agents.base")


class BaseAgent:
    """
    Standard base class representing a single-agent cognitive module with LangChain Agentic RAG.
    Supports tool-calling execution, local vector corpus retrieval, optional web search,
    and fallback to the centralized LLM inference gateway.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        enable_web_search: bool = False,
        custom_tools: Optional[List[BaseTool]] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.enable_web_search = enable_web_search
        self.tools: List[BaseTool] = (
            custom_tools
            if custom_tools is not None
            else get_agent_tools(include_web_search=enable_web_search)
        )
        logger.info(
            f"Initializing BaseAgent: model={model_name}, temp={temperature}, "
            f"web_search={enable_web_search}, tools={[t.name for t in self.tools]}"
        )

    def get_chat_model(self) -> Any:
        """
        Instantiate LangChain ChatModel corresponding to the active configuration.
        """
        if cognitive_settings.llm_provider == "azure":
            return AzureChatOpenAI(
                azure_endpoint=cognitive_settings.azure_openai_endpoint,
                api_key=cognitive_settings.azure_openai_key,
                api_version=cognitive_settings.azure_openai_api_version,
                azure_deployment=cognitive_settings.azure_openai_deployment,
                temperature=self.temperature,
            )

        # Ollama local inference via OpenAI-compatible /v1 endpoint
        return ChatOpenAI(
            base_url=f"{cognitive_settings.ollama_base_url.rstrip('/')}/v1",
            api_key="ollama",
            model=self.model_name,
            temperature=self.temperature,
            timeout=cognitive_settings.ollama_request_timeout,
        )

    async def invoke_agentic_rag(
        self,
        prompt: str,
        system_instruction: str = "",
        tools: Optional[List[BaseTool]] = None,
    ) -> str:
        """
        Invokes LangChain Agentic RAG loop using ReAct tool-calling graph.
        Autonomously retrieves knowledge corpus passages and web information (if enabled)
        before formulating the final response.

        Args:
            prompt: User-facing prompt or pedagogical task instruction.
            system_instruction: High-priority system prompt establishing role and rules.
            tools: Optional custom tools list overriding default agent tools.

        Returns:
            Sanitized text output from the agentic reasoning graph.
        """
        active_tools = tools if tools is not None else self.tools
        try:
            model = self.get_chat_model()
            agent_graph = create_agent(
                model=model,
                tools=active_tools,
                system_prompt=system_instruction or "You are an expert cognitive academic assistant.",
            )

            result = await agent_graph.ainvoke({"messages": [HumanMessage(content=prompt)]})
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                content = getattr(last_msg, "content", "")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                elif isinstance(content, list):
                    text_parts = [
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    ]
                    combined = "".join(text_parts).strip()
                    if combined:
                        return combined

            logger.warning("Agentic RAG returned empty message, falling back to direct chat invocation.")
            return await self.invoke_chat(prompt=prompt, system_instruction=system_instruction)

        except Exception as exc:
            logger.warning(
                f"Agentic RAG execution failed ({exc}), falling back to direct LLM gateway chat invocation."
            )
            return await self.invoke_chat(prompt=prompt, system_instruction=system_instruction)

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
