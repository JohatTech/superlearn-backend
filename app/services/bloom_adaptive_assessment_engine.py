"""
===============================================================================
BLOOM'S TAXONOMY ADAPTIVE ASSESSMENT & ANTI-SPOILER EVALUATION ENGINE
===============================================================================

Architectural Role:
-------------------
Synthesizes high-tier Bloom's Taxonomy evaluation questions on-the-fly, 
manages ephemeral zero-spoiler server-side session state locks, and conducts 
multi-criteria qualitative LLM evaluation upon forced-generation answer submission
with full multi-classroom and test session persistence in Supabase PostgreSQL.

Pedagogical Principles:
-----------------------
- Anti-spoiler payload delivery (zero rubric leaks before submission).
- Forced-generation answer evaluation and qualitative rubric scoring.
- FSRS memory stability updating and Confusion Compass discrepancy diffing.
"""

from __future__ import annotations
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.agents import test_generator_agent, evaluator_agent
from app.models.cognitive_domain_models import ConceptEntity, AdaptiveTestSessionEntity
from app.services.knowledge_graph_engine import knowledge_graph_engine
from app.services.fsrs_spaced_retrieval_scheduler import fsrs_scheduler
from app.services.confusion_compass_diff_engine import confusion_compass_engine

logger = logging.getLogger("superlearn.assessment_engine")

# -----------------------------------------------------------------------------
# Ephemeral In-Memory Session Cache (Zero-Spoiler Protocol)
# Maps session_id -> { concept_id, classroom_id, concept_name, bloom_tier, question, started_at }
# -----------------------------------------------------------------------------
_active_anti_spoiler_sessions: dict[str, dict[str, Any]] = {}

BLOOM_TIER_LABELS: dict[int, str] = {
    1: "Remember",
    2: "Understand",
    3: "Apply",
    4: "Analyze",
    5: "Evaluate",
    6: "Create",
}


class BloomAdaptiveAssessmentEngine:
    """
    Manages question synthesis, anti-spoiler state locks, and qualitative 
    rubric evaluation for high-tier Bloom's taxonomy tests per classroom.
    """

    async def generate_assessment_question(
        self,
        db_session: AsyncSession,
        concept_id: str,
        classroom_id: Optional[str] = None,
        bloom_tier: int = 4,
    ) -> dict[str, Any]:
        """
        Generate a challenging Bloom's question and lock session server-side.
        Returns ONLY public payload: { session_id, question, bloom_tier, bloom_label, concept_name }.
        """
        concept = await db_session.get(ConceptEntity, concept_id)
        if not concept:
            raise ValueError(f"Concept '{concept_id}' not found in database.")

        target_classroom_id = classroom_id or concept.classroom_id
        bloom_tier_clamped = max(4, min(6, bloom_tier))
        bloom_label = BLOOM_TIER_LABELS.get(bloom_tier_clamped, "Analyze")

        question_data = await test_generator_agent.generate_question(
            concept_name=concept.name,
            concept_description=concept.description or "Core domain concept.",
            bloom_tier=bloom_tier_clamped,
        )
        question_text = question_data["question"]

        session_id = str(uuid.uuid4())
        session_entity = AdaptiveTestSessionEntity(
            id=session_id,
            classroom_id=target_classroom_id,
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
            "classroom_id": target_classroom_id,
            "concept_name": concept.name,
            "bloom_tier": bloom_tier_clamped,
            "question_prompt": question_text,
            "started_at": datetime.now(timezone.utc),
        }

        logger.info(
            f"Generated Bloom L{bloom_tier_clamped} test session {session_id} for '{concept.name}' (Classroom: {target_classroom_id})."
        )

        return {
            "session_id": session_id,
            "question": question_text,
            "bloom_tier": bloom_tier_clamped,
            "bloom_label": bloom_label,
            "concept_name": concept.name,
            "classroom_id": target_classroom_id,
        }

    async def evaluate_submitted_answer(
        self,
        db_session: AsyncSession,
        session_id: str,
        student_answer: str,
        effort_latency_seconds: int,
    ) -> dict[str, Any]:
        """
        Evaluate student forced-generation response, persist full test score & feedback,
        update FSRS stability, and return full cognitive feedback and Confusion Compass links.
        """
        session_data = _active_anti_spoiler_sessions.get(session_id)
        if not session_data:
            session_entity = await db_session.get(AdaptiveTestSessionEntity, session_id)
            if not session_entity or session_entity.is_submitted:
                raise ValueError(f"Test session '{session_id}' is invalid or already submitted.")
            
            concept = await db_session.get(ConceptEntity, session_entity.concept_id) if session_entity.concept_id else None
            session_data = {
                "concept_id": session_entity.concept_id,
                "classroom_id": session_entity.classroom_id,
                "concept_name": concept.name if concept else "",
                "bloom_tier": session_entity.bloom_tier,
                "question_prompt": session_entity.generated_question_prompt,
            }

        concept_name = session_data["concept_name"]
        bloom_tier = session_data["bloom_tier"]
        question_prompt = session_data["question_prompt"]
        concept_id = session_data["concept_id"]
        classroom_id = session_data.get("classroom_id")

        evaluation_payload = await evaluator_agent.evaluate_answer(
            concept_name=concept_name,
            question_prompt=question_prompt,
            student_answer=student_answer,
            bloom_tier=bloom_tier,
        )

        score = float(max(0.0, min(1.0, evaluation_payload.get("score", 0.50))))
        misconceptions = evaluation_payload.get("misconceptions", [])
        strengths = evaluation_payload.get("strengths", [])
        suggested_review = evaluation_payload.get("suggested_review_concepts", [])

        # 1. Update Database Test Session Entity
        db_session_record = await db_session.get(AdaptiveTestSessionEntity, session_id)
        if db_session_record:
            db_session_record.student_answer_text = student_answer
            db_session_record.evaluation_score = score
            db_session_record.qualitative_feedback = evaluation_payload.get("feedback", "")
            db_session_record.detected_misconceptions_json = json.dumps(misconceptions)
            db_session_record.strengths_json = json.dumps(strengths)
            db_session_record.suggested_review_json = json.dumps(suggested_review)
            db_session_record.effort_latency_seconds = effort_latency_seconds
            db_session_record.is_submitted = True
            db_session_record.session_submitted_at = datetime.now(timezone.utc)

        # 2. Update FSRS Spaced Memory Stability Parameters
        fsrs_update = await fsrs_scheduler.update_stability_post_assessment(
            db_session=db_session,
            concept_id=concept_id,
            evaluation_score=score,
            effort_latency_seconds=effort_latency_seconds,
            classroom_id=classroom_id,
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
        discrepancy_matrix = await confusion_compass_engine.compute_graph_discrepancy_matrix(
            db_session=db_session, classroom_id=classroom_id
        )

        # Clean session memory lock
        _active_anti_spoiler_sessions.pop(session_id, None)

        return {
            "session_id": session_id,
            "classroom_id": classroom_id,
            "concept_id": concept_id,
            "score": score,
            "feedback": evaluation_payload.get("feedback", ""),
            "strengths": strengths,
            "misconceptions": misconceptions,
            "suggested_review": suggested_review,
            "bloom_level_demonstrated": evaluation_payload.get("bloom_level_demonstrated", bloom_tier),
            "fsrs_update": fsrs_update,
            "confusion_compass": discrepancy_matrix,
        }


# Global Singleton Assessment Engine
bloom_assessment_engine: BloomAdaptiveAssessmentEngine = BloomAdaptiveAssessmentEngine()
