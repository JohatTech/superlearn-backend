"""
===============================================================================
AGENTS MODULE REGISTRY
===============================================================================
"""

from app.agents.syllabus_agent import SyllabusAgent
from app.agents.test_generator_agent import TestGeneratorAgent
from app.agents.evaluator_agent import EvaluatorAgent
from app.agents.study_materials_agent import StudyMaterialsAgent

# Instantiated singletons for global reuse
syllabus_agent: SyllabusAgent = SyllabusAgent()
test_generator_agent: TestGeneratorAgent = TestGeneratorAgent()
evaluator_agent: EvaluatorAgent = EvaluatorAgent()
study_materials_agent: StudyMaterialsAgent = StudyMaterialsAgent()

__all__ = [
    "syllabus_agent",
    "test_generator_agent",
    "evaluator_agent",
    "study_materials_agent",
]
