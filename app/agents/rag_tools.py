"""
===============================================================================
AGENTIC RAG & WEB SEARCH TOOLS MODULE
===============================================================================

Provides LangChain tools for Agentic RAG workflows:
1. retrieve_knowledge_corpus: Queries the local embedded Qdrant vector database
   for canonical textbook passages, course notes, and domain definitions.
2. search_web: Queries live web search (via DuckDuckGo) for external academic
   syllabi, recent textbooks, seminal papers, tutorials, and online resources.
   RESTRICTION: Only available to SyllabusAgent and StudyMaterialsAgent.
"""

from __future__ import annotations
import asyncio
import concurrent.futures
import logging
from typing import Any, List
from langchain_core.tools import BaseTool, StructuredTool
from app.core.cognitive_config import cognitive_settings

logger = logging.getLogger("superlearn.agents.tools")


async def _async_retrieve_knowledge_corpus(query: str, top_k: int = 4) -> str:
    """Execute asynchronous semantic retrieval against the embedded Qdrant corpus."""
    try:
        from app.services.multisource_contrast_ingestion_engine import contrast_engine
        results = await contrast_engine.search_semantic_passages(query=query, top_k=top_k)
        if not results:
            return f"No relevant passages found in the knowledge corpus for the query: '{query}'."

        passages: list[str] = []
        for idx, item in enumerate(results, 1):
            source = item.get("source_name", "Unknown Source")
            chunk = item.get("chunk_index", 0)
            score = item.get("similarity_score", 0.0)
            text = item.get("passage_text", "").strip()
            passages.append(
                f"[{idx}] Source: {source} (Chunk {chunk}, Relevance: {score:.2f})\n{text}"
            )
        return "\n\n---\n\n".join(passages)
    except Exception as exc:
        logger.warning(f"Error querying knowledge corpus for '{query}': {exc}")
        return f"Knowledge corpus retrieval unavailable or empty for query: '{query}'."


def _sync_retrieve_knowledge_corpus(query: str, top_k: int = 4) -> str:
    """Synchronous wrapper for knowledge corpus retrieval."""
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _async_retrieve_knowledge_corpus(query, top_k))
                return future.result(timeout=10.0)
        return asyncio.run(_async_retrieve_knowledge_corpus(query, top_k))
    except Exception as exc:
        logger.warning(f"Sync knowledge corpus retrieval failed: {exc}")
        return f"Knowledge corpus retrieval unavailable for query: '{query}'."


knowledge_corpus_tool: BaseTool = StructuredTool.from_function(
    func=_sync_retrieve_knowledge_corpus,
    coroutine=_async_retrieve_knowledge_corpus,
    name="retrieve_knowledge_corpus",
    description=(
        "Search the local knowledge corpus and indexed course materials for canonical "
        "textbook passages, definitions, and technical explanations."
    ),
)


def _sync_web_search(query: str) -> str:
    """Synchronous web search using DuckDuckGo."""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        searcher = DuckDuckGoSearchRun()
        return searcher.invoke(query)
    except Exception as exc:
        logger.warning(f"Sync web search failed for '{query}': {exc}")
        return f"Web search could not retrieve external results for '{query}' (Network/Rate limit)."


async def _async_web_search(query: str) -> str:
    """Asynchronous web search using DuckDuckGo."""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        searcher = DuckDuckGoSearchRun()
        return await searcher.ainvoke(query)
    except Exception as exc:
        logger.warning(f"Async web search failed for '{query}': {exc}")
        return f"Web search could not retrieve external results for '{query}' (Network/Rate limit)."


web_search_tool: BaseTool = StructuredTool.from_function(
    func=_sync_web_search,
    coroutine=_async_web_search,
    name="search_web",
    description=(
        "Search the web for academic curricula, university syllabi, textbooks, "
        "research papers, online tutorials, and educational resources."
    ),
)


def get_agent_tools(include_web_search: bool = False) -> List[BaseTool]:
    """
    Constructs the tool suite for an agent.

    Args:
        include_web_search: If True, adds the search_web tool to the agent.
                            Strictly reserved for SyllabusAgent and StudyMaterialsAgent.

    Returns:
        List of configured LangChain BaseTool instances.
    """
    tools: List[BaseTool] = [knowledge_corpus_tool]
    if include_web_search:
        tools.append(web_search_tool)
    return tools
