"""
===============================================================================
BLOOM'S TAXONOMY ADAPTIVE ASSESSMENT & ANTI-SPOILER EVALUATION ENGINE
===============================================================================

Architectural Role:
-------------------
Synthesizes high-tier Bloom's Taxonomy evaluation questions on-the-fly, 
manages ephemeral zero-spoiler server-side session state locks, and conducts 
multi-criteria qualitative LLM evaluation upon forced-generation answer submission.

Cognitive Pedagogical Framework:
--------------------------------
1. High-Tier Bloom's Taxonomy Targets (Avoiding Lower-Tier Rote Memorization):
   - Tier 4: Analyze (Deconstructing architectures, isolating root causes of system failure).
   - Tier 5: Evaluate (Defending tradeoffs between competing engineering or theoretical paradigms).
   - Tier 6: Create (Synthesizing novel schemas or formulating hypotheses under constraints).

2. Anti-Spoiler Server-Side State Machine (Zero-Spoiler Protocol):
   To prevent inspect-element leaks or passive cognitive anchoring, the generation 
   endpoint returns strictly { session_id, prompt_text, bloom_tier }.
   All ground truth evaluation rubrics, edge conflict mappings, and expected responses 
   remain strictly server-side until the student commits a timestamped text response.

3. Testing Effect & Forced Generation (Roediger & Karpicke, 2006):
   Active generation of an explanation before seeing answers produces significantly 
   greater neural encoding and schema durability than passive recognition.

Diagram: Anti-Spoiler Assessment Lifecycle
------------------------------------------
```
 [Student Client]                                [FastAPI Engine]                 [LLM Gateway]
        |                                                |                              |
        |--- 1. POST /test/generate (Concept ID) ------->|                              |
        |                                                |--- 2. Synthesize Prompt ---->|
        |                                                |<-- 3. High-Tier Question ----|
        |                                                |                              |
        |<-- 4. { session_id, question_text } (LOCKED) --| (Store session in Memory/DB)
        |                                                |
        | [Student Formulates Forced Generation Answer]  |
        |                                                |
        |--- 5. POST /test/submit (answer, latency) ---->|
        |                                                |--- 6. Multi-Criteria Rubric->|
        |                                                |<-- 7. Score & Misconceptions-|
        |                                                |                              |
        |                                                |--- 8. Update FSRS State ---->|
        |                                                |--- 9. Diff Confusion Compass>|
        |                                                |                              |
        |<-- 10. Complete Feedback & Conflict Links -----|                              |
```

Academic Citations:
-------------------
- Bloom, B. S., et al. (1956). "Taxonomy of Educational Objectives: The 
  Classification of Educational Goals. Handbook I: Cognitive Domain." David McKay.
- Anderson, L. W., & Krathwohl, D. R. (2001). "A Taxonomy for Learning, Teaching, 
  and Assessing: A Revision of Bloom's Taxonomy of Educational Objectives." Longman.
- Roediger, H. L., & Karpicke, J. D. (2006). "Test-enhanced learning: Taking memory 
  tests improves long-term retention." Psychological Science, 17(3), 249-255.
"""

from __future__ import annotations
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.llm_inference_gateway import llm_gateway
from app.models.cognitive_domain_models import ConceptEntity, AdaptiveTestSessionEntity
from app.services.knowledge_graph_engine import knowledge_graph_engine
from app.services.fsrs_spaced_retrieval_scheduler import fsrs_scheduler
from app.services.confusion_compass_diff_engine import confusion_compass_engine

logger = logging.getLogger("superlearn.assessment_engine")

# -----------------------------------------------------------------------------
# Ephemeral In-Memory Session Cache (Zero-Spoiler Protocol)
# Maps session_id -> { concept_id, concept_name, bloom_tier, question, started_at }
# -----------------------------------------------------------------------------
_active_anti_spoiler_sessions: dict[str, dict[str, Any]] = {}

BLOOM_TIER_LABELS: dict[int, str] = {
    4: "Analyze (Deconstruct & Relate)",
    5: "Evaluate (Critique & Justify)",
    6: "Create (Synthesize & Formulate)",
}

QUESTION_SYNTHESIS_SYSTEM_PROMPT = """You are an elite cognitive psychology professor designing rigorous university-level exam questions.
Your goal is to test deep conceptual understanding and transfer, NOT rote memorization.
You must respond ONLY with a valid JSON object in the exact specified schema."""

QUESTION_SYNTHESIS_TEMPLATE = """Synthesize a Bloom's Taxonomy Level {bloom_tier} ({bloom_label}) assessment question for the concept: "{concept_name}".

Context / Prerequisite Information:
{concept_description}

Pedagogical Directives:
1. Target Level {bloom_tier}: Require deconstruction, trade-off evaluation, or architectural design.
2. The question must require a 2-4 paragraph explanatory synthesis.
3. Completely self-contained.

Respond ONLY with this JSON structure:
{{
  "question": "...",
  "bloom_tier": {bloom_tier},
  "expected_synthesis_scope": "2-4 paragraphs"
}}"""

EVALUATION_SYSTEM_PROMPT = """You are a rigorous cognitive assessment rubric evaluator.
Analyze the student's answer for conceptual correctness, depth of mechanistic explanation, and subtle misconceptions.
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


class BloomAdaptiveAssessmentEngine:
    """
    Manages question synthesis, anti-spoiler state locks, and qualitative 
    rubric evaluation for high-tier Bloom's taxonomy tests.
    """

    async def generate_assessment_question(
        self,
        db_session: AsyncSession,
        concept_id: str,
        bloom_tier: int = 4,
    ) -> dict[str, Any]:
        """
        Generate a challenging Bloom's question and lock session server-side.
        Returns ONLY public payload: { session_id, question, bloom_tier }.

        Args:
            db_session: Active database session.
            concept_id: UUID of concept to evaluate.
            bloom_tier: Target Bloom tier (4, 5, or 6).

        Returns:
            Sanitized dictionary safe for client presentation (zero spoiler leaks).
        """
        concept = await db_session.get(ConceptEntity, concept_id)
        if not concept:
            raise ValueError(f"Concept '{concept_id}' not found in database.")

        bloom_tier_clamped = max(4, min(6, bloom_tier))
        bloom_label = BLOOM_TIER_LABELS.get(bloom_tier_clamped, "Analyze")

        prompt = QUESTION_SYNTHESIS_TEMPLATE.format(
            bloom_tier=bloom_tier_clamped,
            bloom_label=bloom_label,
            concept_name=concept.name,
            concept_description=concept.description or "Core domain concept.",
        )

        raw_llm_response = await llm_gateway.generate_response(
            prompt=prompt,
            system_instruction=QUESTION_SYNTHESIS_SYSTEM_PROMPT,
            temperature=0.7,
        )

        try:
            clean_json_str = (
                raw_llm_response.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            parsed = json.loads(clean_json_str)
            question_text = parsed.get("question", raw_llm_response.strip())
        except (json.JSONDecodeError, KeyError):
            question_text = raw_llm_response.strip()

        session_id = str(uuid.uuid4())
        session_entity = AdaptiveTestSessionEntity(
            id=session_id,
            concept_id=concept_id,
            bloom_tier=bloom_tier_clamped,
            generated_question_prompt=question_text,
            is_submitted=False,
        )
        db_session.add(session_entity)
        await db_session.flush()

        # Cache in memory for fast anti-spoiler lookup
        _active_anti_spoiler_sessions[session_id] = {
            "concept_id": concept_id,
            "concept_name": concept.name,
            "bloom_tier": bloom_tier_clamped,
            "question_prompt": question_text,
            "started_at": datetime.now(timezone.utc),
        }

        logger.info(f"Generated Bloom L{bloom_tier_clamped} test session {session_id} for '{concept.name}'.")

        return {
            "session_id": session_id,
            "question": question_text,
            "bloom_tier": bloom_tier_clamped,
            "bloom_label": bloom_label,
            "concept_name": concept.name,
        }

    async def evaluate_submitted_answer(
        self,
        db_session: AsyncSession,
        session_id: str,
        student_answer: str,
        effort_latency_seconds: int,
    ) -> dict[str, Any]:
        """
        Evaluate student forced-generation response, update FSRS stability, 
        and return full cognitive feedback and Confusion Compass links.

        Args:
            db_session: Active database session.
            session_id: UUID of active test session.
            student_answer: Student's submitted text explanation.
            effort_latency_seconds: Recorded duration in seconds.

        Returns:
            Comprehensive evaluation payload with rubric score, feedback, and graph diffs.
        """
        session_data = _active_anti_spoiler_sessions.get(session_id)
        if not session_data:
            session_entity = await db_session.get(AdaptiveTestSessionEntity, session_id)
            if not session_entity or session_entity.is_submitted:
                raise ValueError(f"Test session '{session_id}' is invalid or already submitted.")
            session_data = {
                "concept_id": session_entity.concept_id,
                "concept_name": "",
                "bloom_tier": session_entity.bloom_tier,
                "question_prompt": session_entity.generated_question_prompt,
            }

        concept_name = session_data["concept_name"]
        bloom_tier = session_data["bloom_tier"]
        question_prompt = session_data["question_prompt"]
        concept_id = session_data["concept_id"]

        eval_prompt = EVALUATION_TEMPLATE.format(
            concept_name=concept_name,
            bloom_tier=bloom_tier,
            question_prompt=question_prompt,
            student_answer=student_answer,
        )

        raw_eval_response = await llm_gateway.generate_response(
            prompt=eval_prompt,
            system_instruction=EVALUATION_SYSTEM_PROMPT,
            temperature=0.3,
        )

        try:
            clean_json = (
                raw_eval_response.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            evaluation_payload = json.loads(clean_json)
        except (json.JSONDecodeError, KeyError):
            evaluation_payload = {
                "score": 0.50,
                "bloom_level_demonstrated": bloom_tier,
                "feedback": raw_eval_response.strip(),
                "strengths": [],
                "misconceptions": [],
                "suggested_review_concepts": [],
            }

        score = float(max(0.0, min(1.0, evaluation_payload.get("score", 0.50))))
        misconceptions = evaluation_payload.get("misconceptions", [])

        # 1. Update Database Test Session Entity
        db_session_record = await db_session.get(AdaptiveTestSessionEntity, session_id)
        if db_session_record:
            db_session_record.student_answer_text = student_answer
            db_session_record.evaluation_score = score
            db_session_record.qualitative_feedback = evaluation_payload.get("feedback", "")
            db_session_record.detected_misconceptions_json = json.dumps(misconceptions)
            db_session_record.effort_latency_seconds = effort_latency_seconds
            db_session_record.is_submitted = True
            db_session_record.session_submitted_at = datetime.now(timezone.utc)

        # 2. Update FSRS Spaced Memory Stability Parameters
        fsrs_update = fsrs_scheduler.update_stability_post_assessment(
            concept_id=concept_id,
            evaluation_score=score,
            effort_latency_seconds=effort_latency_seconds,
        )

        # 3. Update In-Memory and DB Concept Mastery State
        await knowledge_graph_engine.update_concept_mastery(
            db_session=db_session, concept_id=concept_id, updated_mastery=score
        )

        # Update Concept FSRS fields in DB
        concept_entity = await db_session.get(ConceptEntity, concept_id)
        if concept_entity:
            concept_entity.fsrs_stability = fsrs_update["new_stability"]
            concept_entity.fsrs_difficulty = fsrs_update["new_difficulty"]
            concept_entity.total_review_count += 1
            concept_entity.last_retrieval_timestamp = datetime.now(timezone.utc)

        # 4. Compute Confusion Compass Discrepancies
        discrepancy_matrix = confusion_compass_engine.compute_graph_discrepancy_matrix()

        # Clean session memory lock
        _active_anti_spoiler_sessions.pop(session_id, None)

        return {
            "session_id": session_id,
            "score": score,
            "feedback": evaluation_payload.get("feedback", ""),
            "strengths": evaluation_payload.get("strengths", []),
            "misconceptions": misconceptions,
            "suggested_review": evaluation_payload.get("suggested_review_concepts", []),
            "bloom_level_demonstrated": evaluation_payload.get("bloom_level_demonstrated", bloom_tier),
            "fsrs_update": fsrs_update,
            "confusion_compass": discrepancy_matrix,
        }


# Global Singleton Assessment Engine
bloom_assessment_engine: BloomAdaptiveAssessmentEngine = BloomAdaptiveAssessmentEngine()
