"""
===============================================================================
BLOOM'S ADAPTIVE ASSESSMENT & ANTI-SPOILER PORTAL ROUTER
===============================================================================

Exposes REST endpoints for:
- Ephemeral test session initialization & Bloom's question synthesis per classroom.
- Anti-spoiler payload delivery (zero rubric leaks before submission).
- Forced-generation answer evaluation, qualitative rubric feedback, and FSRS updating.
- Historical test session retrieval and classroom score analytics.
"""

from __future__ import annotations
import json
from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.supabase_persistence_manager import get_db_session
from app.models.cognitive_domain_models import AdaptiveTestSessionEntity, ConceptEntity, ClassroomEntity
from app.services.bloom_adaptive_assessment_engine import bloom_assessment_engine
from app.services.fsrs_spaced_retrieval_scheduler import fsrs_scheduler

router = APIRouter(prefix="/api/v1/test", tags=["Adaptive Testing & Bloom Assessment"])


class GenerateAssessmentRequestSchema(BaseModel):
    """Schema for requesting a new Bloom's taxonomy test question."""
    concept_id: str = Field(..., description="UUID of target concept to evaluate")
    classroom_id: Optional[str] = Field(default=None, description="Optional Classroom UUID")
    bloom_tier: int = Field(
        default=4, ge=4, le=6, description="Bloom taxonomy tier: 4 (Analyze), 5 (Evaluate), 6 (Create)"
    )


class SubmitAnswerRequestSchema(BaseModel):
    """Schema for submitting forced-generation answer for evaluation."""
    session_id: str = Field(..., description="UUID of active ephemeral test session")
    answer_text: str = Field(..., min_length=5, description="Student's explanatory text answer")
    effort_latency_seconds: int = Field(
        default=0, ge=0, description="Elapsed latency in seconds from prompt to submission"
    )


@router.post(
    "/generate",
    status_code=status.HTTP_201_CREATED,
    summary="Generate Bloom's Taxonomy Question (Anti-Spoiler Locked)",
    description="Synthesizes a high-tier question on-the-fly and locks the session in Supabase. Returns question and session_id ONLY.",
)
async def generate_bloom_question(
    payload: GenerateAssessmentRequestSchema,
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Generate question with anti-spoiler server-side lock."""
    try:
        session_payload = await bloom_assessment_engine.generate_assessment_question(
            db_session=db_session,
            concept_id=payload.concept_id,
            classroom_id=payload.classroom_id,
            bloom_tier=payload.bloom_tier,
        )
        return session_payload
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate assessment: {exc}",
        )


@router.post(
    "/submit",
    summary="Submit Answer for Qualitative Rubric Evaluation",
    description="Submits student's forced generation answer. Evaluates conceptual depth, detects misconceptions, updates FSRS, and persists results to Supabase.",
)
async def submit_assessment_answer(
    payload: SubmitAnswerRequestSchema,
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Submit answer attempt, unlock feedback, and update memory stability."""
    if not payload.answer_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Answer cannot be blank. Forced generation requires an explanation.",
        )

    try:
        evaluation_result = await bloom_assessment_engine.evaluate_submitted_answer(
            db_session=db_session,
            session_id=payload.session_id,
            student_answer=payload.answer_text.strip(),
            effort_latency_seconds=payload.effort_latency_seconds,
        )
        return evaluation_result
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Assessment evaluation failed: {exc}",
        )


@router.get(
    "/history",
    summary="Get Test Sessions History",
    description="Retrieves a historical log of submitted test sessions and scores, filterable by classroom or concept.",
)
async def get_test_history(
    classroom_id: Optional[str] = Query(default=None, description="Optional classroom UUID filter"),
    concept_id: Optional[str] = Query(default=None, description="Optional concept UUID filter"),
    limit: int = Query(default=50, ge=1, le=200, description="Max history items"),
    db_session: AsyncSession = Depends(get_db_session),
) -> List[dict[str, Any]]:
    """Retrieve test history and scores from database."""
    try:
        stmt = select(AdaptiveTestSessionEntity).where(AdaptiveTestSessionEntity.is_submitted.is_(True))

        if classroom_id:
            stmt = stmt.where(AdaptiveTestSessionEntity.classroom_id == classroom_id)
        if concept_id:
            stmt = stmt.where(AdaptiveTestSessionEntity.concept_id == concept_id)

        stmt = stmt.order_by(AdaptiveTestSessionEntity.session_submitted_at.desc()).limit(limit)
        result = await db_session.execute(stmt)
        sessions = result.scalars().all()

        history_items = []
        for s in sessions:
            concept_name = ""
            if s.concept_id:
                concept = await db_session.get(ConceptEntity, s.concept_id)
                if concept:
                    concept_name = concept.name

            misconceptions = []
            if s.detected_misconceptions_json:
                try:
                    misconceptions = json.loads(s.detected_misconceptions_json)
                except Exception:
                    misconceptions = []

            strengths = []
            if s.strengths_json:
                try:
                    strengths = json.loads(s.strengths_json)
                except Exception:
                    strengths = []

            history_items.append({
                "session_id": s.id,
                "classroom_id": s.classroom_id,
                "concept_id": s.concept_id,
                "concept_name": concept_name,
                "bloom_tier": s.bloom_tier,
                "question": s.generated_question_prompt,
                "student_answer": s.student_answer_text,
                "score": s.evaluation_score,
                "feedback": s.qualitative_feedback,
                "misconceptions": misconceptions,
                "strengths": strengths,
                "effort_latency_seconds": s.effort_latency_seconds,
                "submitted_at": s.session_submitted_at.isoformat() if s.session_submitted_at else None,
            })

        return history_items
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch test history: {exc}",
        )


@router.get(
    "/session/{session_id}",
    summary="Get Specific Test Session Details",
    description="Retrieves full question prompt, response, feedback, and scoring for a test session.",
)
async def get_test_session_details(
    session_id: str,
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Retrieve details for a single test session."""
    session = await db_session.get(AdaptiveTestSessionEntity, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test session '{session_id}' not found.",
        )

    concept_name = ""
    if session.concept_id:
        concept = await db_session.get(ConceptEntity, session.concept_id)
        if concept:
            concept_name = concept.name

    misconceptions = []
    if session.detected_misconceptions_json:
        try:
            misconceptions = json.loads(session.detected_misconceptions_json)
        except Exception:
            misconceptions = []

    strengths = []
    if session.strengths_json:
        try:
            strengths = json.loads(session.strengths_json)
        except Exception:
            strengths = []

    return {
        "session_id": session.id,
        "classroom_id": session.classroom_id,
        "concept_id": session.concept_id,
        "concept_name": concept_name,
        "bloom_tier": session.bloom_tier,
        "question": session.generated_question_prompt,
        "student_answer": session.student_answer_text,
        "score": session.evaluation_score,
        "feedback": session.qualitative_feedback,
        "misconceptions": misconceptions,
        "strengths": strengths,
        "effort_latency_seconds": session.effort_latency_seconds,
        "is_submitted": session.is_submitted,
        "started_at": session.session_started_at.isoformat() if session.session_started_at else None,
        "submitted_at": session.session_submitted_at.isoformat() if session.session_submitted_at else None,
    }


@router.get(
    "/classroom/{classroom_id}/analytics",
    summary="Get Classroom Learning Analytics & Score Telemetry",
    description="Aggregates learning mastery, score history over time, and cognitive telemetry for a specific classroom.",
)
async def get_classroom_analytics(
    classroom_id: str,
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Compute aggregate telemetry and analytics for a classroom."""
    classroom = await db_session.get(ClassroomEntity, classroom_id)
    if not classroom:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Classroom '{classroom_id}' not found.",
        )

    # 1. Fetch Concepts for this classroom
    concept_stmt = select(ConceptEntity).where(ConceptEntity.classroom_id == classroom_id)
    concept_records = (await db_session.execute(concept_stmt)).scalars().all()

    total_concepts = len(concept_records)
    mastered_count = sum(1 for c in concept_records if c.mastery_score >= 0.85)
    in_progress_count = sum(1 for c in concept_records if 0.0 < c.mastery_score < 0.85)
    unstarted_count = sum(1 for c in concept_records if c.mastery_score == 0.0)
    avg_mastery = (
        sum(c.mastery_score for c in concept_records) / total_concepts if total_concepts > 0 else 0.0
    )

    # 2. Fetch Submitted Test Sessions for this classroom
    test_stmt = (
        select(AdaptiveTestSessionEntity)
        .where(
            AdaptiveTestSessionEntity.classroom_id == classroom_id,
            AdaptiveTestSessionEntity.is_submitted.is_(True),
        )
        .order_by(AdaptiveTestSessionEntity.session_submitted_at.asc())
    )
    test_records = (await db_session.execute(test_stmt)).scalars().all()

    total_tests = len(test_records)
    scores = [t.evaluation_score for t in test_records if t.evaluation_score is not None]
    avg_score = (sum(scores) / len(scores)) if scores else 0.0

    score_progression = [
        {
            "session_id": t.id,
            "concept_id": t.concept_id,
            "score": round(t.evaluation_score, 3) if t.evaluation_score is not None else 0.0,
            "bloom_tier": t.bloom_tier,
            "submitted_at": t.session_submitted_at.isoformat() if t.session_submitted_at else None,
        }
        for t in test_records
    ]

    # 3. Aggregate Detected Misconceptions Frequency
    misconception_freq: dict[str, int] = {}
    for t in test_records:
        if t.detected_misconceptions_json:
            try:
                for m in json.loads(t.detected_misconceptions_json):
                    if isinstance(m, str) and m.strip():
                        misconception_freq[m] = misconception_freq.get(m, 0) + 1
            except Exception:
                pass

    sorted_misconceptions = [
        {"misconception": m, "count": count}
        for m, count in sorted(misconception_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    ]

    # 4. FSRS recommendations for this classroom
    recommendations = await fsrs_scheduler.compute_priority_recommendations(
        db_session=db_session, classroom_id=classroom_id, top_k=5
    )

    return {
        "classroom_id": classroom.id,
        "classroom_title": classroom.title,
        "total_concepts": total_concepts,
        "mastered_concepts": mastered_count,
        "in_progress_concepts": in_progress_count,
        "unstarted_concepts": unstarted_count,
        "average_concept_mastery": round(avg_mastery, 3),
        "total_tests_completed": total_tests,
        "average_test_score": round(avg_score, 3),
        "score_progression": score_progression,
        "top_misconceptions": sorted_misconceptions,
        "recommendations": recommendations,
    }
