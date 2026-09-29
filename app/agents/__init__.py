"""
===============================================================================
AGENTS MODULE REGISTRY
===============================================================================
"""

from app.agents.base_agent import BaseAgent
from app.agents.syllabus_agent import SyllabusAgent, syllabus_agent
from app.agents.test_generator_agent import TestGeneratorAgent, test_generator_agent
from app.agents.evaluator_agent import EvaluatorAgent, evaluator_agent
from app.agents.study_materials_agent import StudyMaterialsAgent, study_materials_agent
from app.agents.rag_tools import (
    knowledge_corpus_tool,
    web_search_tool,
    get_agent_tools,
)

__all__ = [
    "BaseAgent",
    "SyllabusAgent",
    "syllabus_agent",
    "TestGeneratorAgent",
    "test_generator_agent",
    "EvaluatorAgent",
    "evaluator_agent",
    "StudyMaterialsAgent",
    "study_materials_agent",
    "knowledge_corpus_tool",
    "web_search_tool",
    "get_agent_tools",
]
