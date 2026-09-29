"""
===============================================================================
EVALUATOR & AUDITOR AGENT (COGNITIVE EVALUATOR & GRAPH AUDITOR)
===============================================================================
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List
from app.agents.base_agent import BaseAgent
from app.core.cognitive_config import cognitive_settings

logger = logging.getLogger("superlearn.agents.evaluator")

EVALUATION_SYSTEM_PROMPT = """You are a rigorous cognitive assessment rubric evaluator.
Analyze the student's answer for conceptual correctness, depth of mechanistic explanation, and subtle misconceptions.
You have access to tools:
- retrieve_knowledge_corpus: query local textbooks, course notes, and ground-truth definitions to rigorously verify student statements.
(Note: External web search is disabled for evaluation security).

Respond ONLY with a valid JSON object in the exact specified schema."""

EVALUATION_TEMPLATE = """Evaluate this student answer against Bloom's Taxonomy Level {bloom_tier} standards.

Concept: "{concept_name}"
Question: "{question_prompt}"
Student's Answer Attempt:
"{student_answer}"

Evaluation Rubric:
1. Conceptual Accuracy (Is the fundamental mechanism correct?)
2. Explanatory Depth (Does the student explain *why* and *how*, rather than just stating definitions?)
3. Misconceptions (Identify specific false assumptions or inverted relationships).

Respond ONLY with this JSON structure:
{{
  "score": 0.85,
  "bloom_level_demonstrated": {bloom_tier},
  "feedback": "Detailed paragraph explaining what was correct and what needs refinement...",
  "strengths": ["Clear explanation of X", "Accurate connection to Y"],
  "misconceptions": ["Assumes A causes B when in reality B causes A"],
  "suggested_review_concepts": ["concept_1", "concept_2"]
}}

Note: 'score' must be a continuous float in [0.0, 1.0]."""

GRAPH_AUDIT_SYSTEM_PROMPT = """You are an expert cognitive psychologist and curriculum auditor.
Analyze the provided mental schema graph layout (nodes and connection edges) to detect misconceptions, circular learning loops, logical gaps, or incorrect sequence structures.
You have access to tools:
- retrieve_knowledge_corpus: query canonical prerequisite relationships and definitions from the indexed course materials.

Respond ONLY with a valid JSON object in the exact specified schema."""

GRAPH_AUDIT_TEMPLATE = """Audit this student mental schema graph representation.

Concepts (Nodes):
{concepts}

Connection Prerequisite Edges:
{edges}

Pedagogical Audit Criteria:
1. Circular Dependencies (e.g. Concept A is a prerequisite of B, and B is a prerequisite of A).
2. Logical Leaps (e.g. Moving to an advanced topic without setting up fundamental steps).
3. Incorrect/Inverted prerequisites.

Respond ONLY with this JSON structure:
{{
  "has_warnings": true,
  "warnings": [
    {{
      "type": "circular_dependency / logical_leap / inverted_relation",
      "severity": "high / medium / low",
      "message": "Detailed warning explanation...",
      "affected_nodes": ["node_id_1", "node_id_2"]
    }}
  ],
  "recommendations": [
    "Suggested graph adjustments or concepts to add/fix..."
  ]
}}"""


class EvaluatorAgent(BaseAgent):
    """
    Agent responsible for scoring responses and auditing knowledge graphs.
    Equipped with LangChain Agentic RAG: Vector Knowledge Retrieval (Web Search disabled).
    """

    def __init__(self) -> None:
        super().__init__(
            model_name=cognitive_settings.agent_evaluator_model,
            temperature=0.2,  # Low temperature for strict grading and auditing consistency
            enable_web_search=False,
        )

    async def evaluate_answer(
        self,
        concept_name: str,
        question_prompt: str,
        student_answer: str,
        bloom_tier: int = 4,
    ) -> Dict[str, Any]:
        """
        Evaluate and score a student's answer using the evaluation rubric via Agentic RAG.
        """
        logger.info(f"Evaluator Agent grading answer for '{concept_name}' using model {self.model_name}")
        prompt = EVALUATION_TEMPLATE.format(
            bloom_tier=bloom_tier,
            concept_name=concept_name,
            question_prompt=question_prompt,
            student_answer=student_answer,
        )

        try:
            raw_response = await self.invoke_agentic_rag(
                prompt=prompt,
                system_instruction=EVALUATION_SYSTEM_PROMPT,
            )
            parsed = self.parse_json(raw_response)
            return {
                "score": float(parsed.get("score", 0.5)),
                "bloom_level_demonstrated": int(parsed.get("bloom_level_demonstrated", bloom_tier)),
                "feedback": parsed.get("feedback", "No feedback generated."),
                "strengths": parsed.get("strengths", []),
                "misconceptions": parsed.get("misconceptions", []),
                "suggested_review_concepts": parsed.get("suggested_review_concepts", []),
            }
        except Exception as exc:
            logger.error(f"Evaluator Agent grading failed: {exc}")
            # Fallback safe grade
            return {
                "score": 0.5,
                "bloom_level_demonstrated": bloom_tier,
                "feedback": "Grading fallback due to system parsing error.",
                "strengths": ["Answer submitted"],
                "misconceptions": ["Evaluation parsing fallback error"],
                "suggested_review_concepts": [concept_name],
            }

    async def audit_knowledge_graph(
        self,
        concepts: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute structural and cognitive audit of the student's mental model graph via Agentic RAG.
        """
        logger.info(f"Evaluator Agent auditing knowledge graph ({len(concepts)} nodes, {len(edges)} edges)")

        prompt = GRAPH_AUDIT_TEMPLATE.format(
            concepts=json.dumps(concepts, indent=2),
            edges=json.dumps(edges, indent=2),
        )

        try:
            raw_response = await self.invoke_agentic_rag(
                prompt=prompt,
                system_instruction=GRAPH_AUDIT_SYSTEM_PROMPT,
            )
            return self.parse_json(raw_response)
        except Exception as exc:
            logger.error(f"Evaluator Agent audit parsing failed: {exc}")
            return {
                "has_warnings": False,
                "warnings": [],
                "recommendations": ["No audit recommendations available."],
            }


evaluator_agent: EvaluatorAgent = EvaluatorAgent()
