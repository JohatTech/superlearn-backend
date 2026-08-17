"""
===============================================================================
BLOOM'S ADAPTIVE ASSESSMENT & ANTI-SPOILER PORTAL ROUTER
===============================================================================

Exposes REST endpoints for:
- Ephemeral test session initialization & Bloom's question synthesis.
- Anti-spoiler payload delivery (zero rubric leaks before submission).
- Forced-generation answer evaluation, qualitative rubric feedback, and FSRS updating.
"""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.supabase_persistence_manager import get_db_session
from app.services.bloom_adaptive_assessment_engine import bloom_assessment_engine

router = APIRouter(prefix="/api/v1/test", tags=["Adaptive Testing & Bloom Assessment"])


class GenerateAssessmentRequestSchema(BaseModel):
    """Schema for requesting a new Bloom's taxonomy test question."""
    concept_id: str = Field(..., description="UUID of target concept to evaluate")
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
    description="Synthesizes a high-tier question on-the-fly and locks the session. Returns question and session_id ONLY.",
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
    description="Submits student's forced generation answer. Evaluates conceptual depth, detects misconceptions, updates FSRS, and diffs Confusion Compass.",
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
