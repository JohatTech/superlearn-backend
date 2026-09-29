"""
===============================================================================
TEST GENERATOR AGENT (BLOOM QUESTION SYNTHESIZER)
===============================================================================
"""

from __future__ import annotations
import logging
from typing import Any, Dict
from app.agents.base_agent import BaseAgent
from app.core.cognitive_config import cognitive_settings

logger = logging.getLogger("superlearn.agents.test_generator")

BLOOM_TIER_LABELS: dict[int, str] = {
    1: "Remember (Recall facts & basic concepts)",
    2: "Understand (Explain ideas or concepts)",
    3: "Apply (Use information in new situations)",
    4: "Analyze (Deconstruct & Relate)",
    5: "Evaluate (Critique & Justify)",
    6: "Create (Synthesize & Formulate)",
}

QUESTION_SYNTHESIS_SYSTEM_PROMPT = """You are an elite cognitive psychology professor designing rigorous university-level exam questions.
Your goal is to test deep conceptual understanding and transfer, NOT rote memorization.
You must respond ONLY with a valid JSON object in the exact specified schema."""

QUESTION_SYNTHESIS_TEMPLATE = """Synthesize a Bloom's Taxonomy Level {bloom_tier} ({bloom_label}) assessment question for the concept: "{concept_name}".

Context / Reference Information:
{concept_description}

Pedagogical Directives:
1. Target Level {bloom_tier}: Require deconstruction, trade-off evaluation, or design synthesis matching the Bloom's taxonomy depth.
2. The question must require a 2-4 paragraph explanatory synthesis.
3. Completely self-contained.

Respond ONLY with this JSON structure:
{{
  "question": "...",
  "bloom_tier": {bloom_tier},
  "expected_synthesis_scope": "2-4 paragraphs"
}}"""


class TestGeneratorAgent(BaseAgent):
    """
    Agent responsible for generating rigorous Bloom's taxonomy questions.
    """

    def __init__(self) -> None:
        super().__init__(
            model_name=cognitive_settings.agent_test_gen_model,
            temperature=0.8,  # Slightly higher temperature for creative questions
        )

    async def generate_question(
        self,
        concept_name: str,
        concept_description: str,
        bloom_tier: int = 4,
    ) -> Dict[str, Any]:
        """
        Query LLM to synthesize a specific taxonomy tier question for the given concept.
        """
        logger.info(f"Test Gen Agent synthesizing question for '{concept_name}' (Tier {bloom_tier}) using model {self.model_name}")
        
        bloom_label = BLOOM_TIER_LABELS.get(bloom_tier, "Analyze")
        prompt = QUESTION_SYNTHESIS_TEMPLATE.format(
            bloom_tier=bloom_tier,
            bloom_label=bloom_label,
            concept_name=concept_name,
            concept_description=concept_description or "No reference context available.",
        )

        raw_response = await self.invoke_chat(
            prompt=prompt,
            system_instruction=QUESTION_SYNTHESIS_SYSTEM_PROMPT,
        )

        try:
            parsed = self.parse_json(raw_response)
            return {
                "question": parsed.get("question", f"Explain the main characteristics of {concept_name}."),
                "bloom_tier": parsed.get("bloom_tier", bloom_tier),
                "expected_synthesis_scope": parsed.get("expected_synthesis_scope", "2-4 paragraphs"),
            }
        except Exception as exc:
            logger.error(f"Test Gen Agent parsing failed: {exc}. Raw response: {raw_response}")
            # Fallback
            return {
                "question": f"Explain the key architectural and design principles of '{concept_name}' and deconstruct its operational tradeoffs.",
                "bloom_tier": bloom_tier,
                "expected_synthesis_scope": "2-4 paragraphs",
            }

