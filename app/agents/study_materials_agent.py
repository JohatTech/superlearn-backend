"""
===============================================================================
STUDY MATERIALS RESEARCHER AGENT (BOOK & RESOURCE FINDER)
===============================================================================
"""

from __future__ import annotations
import logging
from typing import Any, Dict
from app.agents.base_agent import BaseAgent
from app.core.cognitive_config import cognitive_settings

logger = logging.getLogger("superlearn.agents.study_materials")

RESEARCH_SYSTEM_PROMPT = """You are an academic librarian and learning resources research agent.
Your goal is to suggest the absolute best study materials, books, research papers, YouTube channels, and online tutorials for a given concept.
You must respond ONLY with a valid JSON object in the exact specified schema."""

RESEARCH_TEMPLATE = """Research and recommend the most effective study materials and reference resources for the following topic:
Concept: "{concept_name}"
Description: "{concept_description}"

Pedagogical Resource Directives:
1. Provide standard university-level textbooks with specific chapters/sections if applicable.
2. Recommend key classical or foundational scientific papers.
3. Suggest online interactive resources (e.g. websites, simulators, open-courseware).
4. List popular media channels or YouTube series that clarify this concept visually.

Respond ONLY with this JSON structure:
{{
  "concept_name": "{concept_name}",
  "textbooks": [
    {{
      "title": "Book Title",
      "author": "Author name",
      "relevance": "Explanation of how this book helps master this concept..."
    }}
  ],
  "papers": [
    {{
      "title": "Paper Title",
      "authors": "Author names",
      "year": "YYYY",
      "url": "http://example.com/paper"
    }}
  ],
  "online_resources": [
    {{
      "name": "Platform or Resource Name",
      "url": "http://example.com/resource",
      "type": "simulation / course / tutorial"
    }}
  ]
}}"""


class StudyMaterialsAgent(BaseAgent):
    """
    Agent responsible for finding high-quality books, papers, and learning resources.
    """

    def __init__(self) -> None:
        super().__init__(
            model_name=cognitive_settings.agent_study_materials_model,
            temperature=0.6,
        )

    async def recommend_study_materials(
        self,
        concept_name: str,
        concept_description: str,
    ) -> Dict[str, Any]:
        """
        Synthesize textbook and material recommendations for a target concept.
        """
        logger.info(f"Study Materials Agent searching resources for '{concept_name}' using model {self.model_name}")
        prompt = RESEARCH_TEMPLATE.format(
            concept_name=concept_name,
            concept_description=concept_description or "No reference context available.",
        )

        raw_response = await self.invoke_chat(
            prompt=prompt,
            system_instruction=RESEARCH_SYSTEM_PROMPT,
        )

        try:
            return self.parse_json(raw_response)
        except Exception as exc:
            logger.error(f"Study Materials Agent parsing failed: {exc}. Raw response: {raw_response}")
            # Fallback structure
            return {
                "concept_name": concept_name,
                "textbooks": [
                    {
                        "title": f"Introduction to {concept_name}",
                        "author": "Standard Academic Literature",
                        "relevance": "Standard textbook covering the fundamental principles."
                    }
                ],
                "papers": [],
                "online_resources": [
                    {
                        "name": "MIT OpenCourseWare",
                        "url": "https://ocw.mit.edu",
                        "type": "course"
                    }
                ]
            }
