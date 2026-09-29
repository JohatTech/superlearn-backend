"""
===============================================================================
SYLLABUS GENERATOR AGENT (ULTRA-FAST MULTI-STAGE CURRICULUM PIPELINE)
===============================================================================
"""

from __future__ import annotations
import asyncio
import json
import logging
from typing import Any, Dict, List
from app.agents.base_agent import BaseAgent
from app.core.cognitive_config import cognitive_settings

logger = logging.getLogger("superlearn.agents.syllabus")

# ===============================================================================
# STAGE 1: CURRICULUM ARCHITECTURE (CONCISE BLUEPRINT)
# ===============================================================================

SYLLABUS_ARCHITECT_SYSTEM_PROMPT = """You are an elite university curriculum architect.
Create a structured university course blueprint for the subject.
You have access to tools:
- retrieve_knowledge_corpus: query local textbooks and ingested course materials.
- search_web: search the web for latest university syllabi, curriculum standards, and topics.
Use your tools if you need to ground your blueprint in authoritative references.

Identify 6 to 8 major academic modules progressing from fundamentals to advanced mastery.
Keep descriptions concise (1 sentence).

Output strictly valid JSON in this schema:
{
  "syllabus_title": "...",
  "description": "...",
  "modules": [
    {"order": 1, "name": "...", "description": "..."}
  ]
}"""

SYLLABUS_ARCHITECT_USER_TEMPLATE = """Subject: "{topic}"
Generate the 6 to 8 module academic architecture JSON."""


# ===============================================================================
# STAGE 2: BATCHED MODULE EXPANSION (TOKEN-EFFICIENT GRAMMAR SAMPLING)
# ===============================================================================

BATCH_MODULE_EXPANSION_SYSTEM_PROMPT = """You are an expert university course designer.
Expand the provided curriculum modules into 3 to 5 distinct university-level lessons each.
For each lesson provide:
- "name": concise lesson title
- "description": 1 concise sentence
- "concepts": list of 2 to 4 key technical terms

Output strictly valid JSON in this schema:
{
  "modules": [
    {
      "order": 1,
      "name": "...",
      "units": [
        {
          "order": 1,
          "name": "...",
          "description": "...",
          "concepts": ["...", "..."]
        }
      ]
    }
  ]
}"""

BATCH_MODULE_EXPANSION_USER_TEMPLATE = """Course: "{topic}"
Expand these {module_count} modules into 3 to 5 lessons each:
{modules_context}"""


class SyllabusAgent(BaseAgent):
    """
    High-Speed Multi-Stage Agent delivering university-grade curricula.
    Equipped with LangChain Agentic RAG: Vector Knowledge Retrieval and Web Search.
    """

    def __init__(self) -> None:
        super().__init__(
            model_name=cognitive_settings.agent_syllabus_model,
            temperature=0.2,
            enable_web_search=True,
        )

    def robust_json_parse(self, raw_text: str) -> dict:
        """
        Extracts valid JSON from response.
        """
        return self.parse_json(raw_text)

    async def _stage1_architect_curriculum(self, topic: str) -> Dict[str, Any]:
        """
        Stage 1: Fast architecture generation (6-8 modules) via Agentic RAG.
        """
        prompt = SYLLABUS_ARCHITECT_USER_TEMPLATE.format(topic=topic)

        try:
            raw_response = await self.invoke_agentic_rag(
                prompt=prompt,
                system_instruction=SYLLABUS_ARCHITECT_SYSTEM_PROMPT,
            )
            parsed = self.robust_json_parse(raw_response)
            modules = parsed.get("modules", [])
            if isinstance(modules, list) and len(modules) > 0:
                return {
                    "syllabus_title": parsed.get("syllabus_title", f"University Curriculum: {topic}"),
                    "description": parsed.get("description", f"Structured curriculum for {topic}."),
                    "modules": modules,
                }
        except Exception as exc:
            logger.warning(f"Stage 1 LLM parse error: {exc}. Using fallback architecture.")

        return {
            "syllabus_title": f"Curriculum: {topic}",
            "description": f"Structured university curriculum for {topic}.",
            "modules": [
                {"order": 1, "name": f"Foundational Principles of {topic}", "description": "Core concepts and definitions."},
                {"order": 2, "name": f"Theoretical & Mathematical Framework of {topic}", "description": "Analytical and theoretical models."},
                {"order": 3, "name": f"Core Methodologies in {topic}", "description": "Key algorithmic and practical mechanisms."},
                {"order": 4, "name": f"Implementation & Systems Engineering in {topic}", "description": "Practical application and development."},
                {"order": 5, "name": f"Advanced Paradigms & Evaluation in {topic}", "description": "State of the art developments and case studies."},
            ],
        }

    async def _stage2_expand_batch(
        self,
        topic: str,
        modules_batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Batched expansion of 3-4 modules via Agentic RAG.
        """
        modules_context_lines = [
            f"- Mod {m.get('order', i+1)}: \"{m.get('name', 'Module')}\" ({m.get('description', '')})"
            for i, m in enumerate(modules_batch)
        ]
        modules_context = "\n".join(modules_context_lines)

        prompt = BATCH_MODULE_EXPANSION_USER_TEMPLATE.format(
            topic=topic,
            module_count=len(modules_batch),
            modules_context=modules_context,
        )

        try:
            raw_response = await self.invoke_agentic_rag(
                prompt=prompt,
                system_instruction=BATCH_MODULE_EXPANSION_SYSTEM_PROMPT,
            )
            parsed = self.robust_json_parse(raw_response)
            parsed_modules = parsed.get("modules", [])

            if isinstance(parsed_modules, list) and len(parsed_modules) > 0:
                matched_results: List[Dict[str, Any]] = []
                for orig_mod in modules_batch:
                    orig_order = orig_mod.get("order")
                    orig_name = orig_mod.get("name", "")

                    match = next(
                        (
                            p for p in parsed_modules
                            if p.get("order") == orig_order or p.get("name", "").lower() == orig_name.lower()
                        ),
                        None,
                    )

                    if match and match.get("units"):
                        matched_results.append({
                            "order": orig_order,
                            "name": orig_name,
                            "description": orig_mod.get("description", ""),
                            "units": match["units"],
                        })
                    else:
                        matched_results.append(self._generate_fallback_module(orig_mod))
                return matched_results
        except Exception as exc:
            logger.warning(f"Stage 2 batch expansion error: {exc}. Using fallback.")

        return [self._generate_fallback_module(m) for m in modules_batch]

    def _generate_fallback_module(self, mod_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generates concise structured fallback units."""
        mod_order = mod_data.get("order", 1)
        mod_name = mod_data.get("name", f"Module {mod_order}")
        mod_desc = mod_data.get("description", "")
        return {
            "order": mod_order,
            "name": mod_name,
            "description": mod_desc,
            "units": [
                {
                    "order": 1,
                    "name": f"Foundations of {mod_name}",
                    "description": f"Core principles governing {mod_name}.",
                    "concepts": [mod_name, "Foundations"],
                },
                {
                    "order": 2,
                    "name": f"Mechanisms & Theory of {mod_name}",
                    "description": f"Functional breakdown of {mod_name}.",
                    "concepts": [mod_name, "Mechanics"],
                },
                {
                    "order": 3,
                    "name": f"Applied Engineering of {mod_name}",
                    "description": f"Implementation and design patterns for {mod_name}.",
                    "concepts": [mod_name, "Applications"],
                },
            ],
        }

    async def generate_syllabus(self, topic: str) -> Dict[str, Any]:
        """
        Executes multi-stage syllabus generation:
        1. Architecture Blueprint (6-8 modules)
        2. Batched Parallel Expansion (3-4 modules per call)
        3. Flattened aggregation and concept enrichment
        """
        logger.info(f"Starting syllabus generation for '{topic}'")

        # ── Stage 1: Curriculum Architecture ─────────────────────────────────
        arch = await self._stage1_architect_curriculum(topic=topic)
        syllabus_title = arch["syllabus_title"]
        description = arch["description"]
        raw_modules = arch["modules"]

        # ── Stage 2: Chunk into Batches ──────────────────────────────────────
        batch_size = max(3, cognitive_settings.agent_syllabus_module_batch_size)
        batches = [raw_modules[i : i + batch_size] for i in range(0, len(raw_modules), batch_size)]

        expanded_modules: List[Dict[str, Any]] = []
        for batch in batches:
            batch_res = await self._stage2_expand_batch(topic=topic, modules_batch=batch)
            expanded_modules.extend(batch_res)

        expanded_modules.sort(key=lambda m: m.get("order", 0))

        # ── Stage 3: Aggregation & Concept Enrichment ────────────────────────
        flat_topics: List[Dict[str, str]] = []

        for mod in expanded_modules:
            mod_name = mod.get("name", "Module")
            units = mod.get("units", [])
            for unit in units:
                unit_name = unit.get("name", "Unit")
                unit_desc = unit.get("description", "")
                concepts = unit.get("concepts", [])

                formatted_name = f"{mod_name}: {unit_name}"
                formatted_desc = unit_desc
                if isinstance(concepts, list) and concepts:
                    concepts_str = ", ".join(str(c) for c in concepts if c)
                    if concepts_str:
                        formatted_desc = f"{unit_desc.rstrip('.')} — Key concepts: {concepts_str}."

                flat_topics.append({
                    "name": formatted_name,
                    "description": formatted_desc,
                })

        logger.info(
            f"Syllabus generation complete: {len(flat_topics)} topics across "
            f"{len(expanded_modules)} modules."
        )

        return {
            "syllabus_title": syllabus_title,
            "description": description,
            "topics": flat_topics,
            "modules": expanded_modules,
        }


syllabus_agent: SyllabusAgent = SyllabusAgent()
