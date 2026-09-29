"""
===============================================================================
SYLLABUS MASTER & CLASSROOM PERSISTENCE ROUTER
===============================================================================
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.db.supabase_persistence_manager import get_db_session
from app.models.cognitive_domain_models import (
    ClassroomEntity,
    SyllabusItemEntity,
    ConceptEntity,
    AdaptiveTestSessionEntity,
    MindmapUploadEntity,
)
from app.services.syllabus_master_engine import syllabus_master_engine
from app.services.knowledge_graph_engine import knowledge_graph_engine
from app.services.syllabus_comparison_service import syllabus_comparison_service

router = APIRouter(prefix="/api/v1", tags=["Syllabus Master & Classrooms"])


class GenerateSyllabusRequestSchema(BaseModel):
    topic: str = Field(..., min_length=1, max_length=255, description="Topic to generate syllabus for")


class TopicItemSchema(BaseModel):
    name: str
    description: str = ""


class ApproveClassroomRequestSchema(BaseModel):
    topic_query: str
    syllabus_title: str
    description: str = ""
    topics: List[TopicItemSchema]


class UpdateClassroomSchema(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None


class AddSyllabusItemSchema(BaseModel):
    name: str
    description: str = ""
    order_index: Optional[int] = None


@router.post(
    "/syllabus-master/generate",
    summary="Generate Syllabus Preview",
    description="Uses LLM to generate a structured university syllabus layout for a given topic.",
)
async def generate_syllabus_preview(
    payload: GenerateSyllabusRequestSchema,
) -> Dict[str, Any]:
    try:
        preview = await syllabus_master_engine.generate_syllabus(topic=payload.topic.strip())
        return preview
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate syllabus: {exc}",
        )


@router.post(
    "/syllabus-master/compare-models",
    summary="Sequential Multi-Model Syllabus Comparison Benchmark",
    description=(
        "Executes sequential syllabus generation across Phi, Qwen, and Azure OpenAI models. "
        "Measures hardware GPU usage, VRAM consumption, and inference latency for side-by-side benchmarking."
    ),
)
async def compare_syllabus_models(
    payload: GenerateSyllabusRequestSchema,
) -> Dict[str, Any]:
    try:
        topic = payload.topic.strip()
        if not topic:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Topic prompt cannot be empty.")
        result = await syllabus_comparison_service.generate_comparison(topic=topic)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-model syllabus comparison benchmark failed: {exc}",
        )


@router.post(
    "/syllabus-master/approve",
    status_code=status.HTTP_201_CREATED,
    summary="Approve Syllabus & Create Classroom",
    description="Saves an approved syllabus as a Classroom with topics, concepts, and DAG prerequisite edges in Supabase.",
)
async def approve_syllabus_classroom(
    payload: ApproveClassroomRequestSchema,
    db_session: AsyncSession = Depends(get_db_session),
) -> Dict[str, Any]:
    try:
        topics_list = [{"name": t.name, "description": t.description} for t in payload.topics]
        classroom = await syllabus_master_engine.approve_classroom(
            db_session=db_session,
            topic_query=payload.topic_query.strip(),
            syllabus_title=payload.syllabus_title.strip(),
            description=payload.description.strip(),
            topics=topics_list,
        )
        return {
            "id": classroom.id,
            "title": classroom.title,
            "description": classroom.description,
            "topic_query": classroom.topic_query,
            "message": "Classroom and curriculum created successfully.",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to approve syllabus and create classroom: {exc}",
        )


@router.get(
    "/classrooms",
    summary="List All Classrooms",
    description="Retrieves a list of all approved classrooms with statistics on topics, tests, and average score.",
)
async def list_classrooms(
    db_session: AsyncSession = Depends(get_db_session),
) -> List[Dict[str, Any]]:
    try:
        stmt = (
            select(ClassroomEntity)
            .options(
                selectinload(ClassroomEntity.syllabus_items),
                selectinload(ClassroomEntity.test_sessions),
            )
            .order_by(ClassroomEntity.created_at.desc())
        )
        result = await db_session.execute(stmt)
        classrooms = result.scalars().all()

        classroom_list = []
        for c in classrooms:
            submitted_tests = [t for t in c.test_sessions if t.is_submitted and t.evaluation_score is not None]
            avg_score = (
                sum(t.evaluation_score for t in submitted_tests) / len(submitted_tests)
                if submitted_tests
                else 0.0
            )

            classroom_list.append({
                "id": c.id,
                "title": c.title,
                "description": c.description,
                "topic_query": c.topic_query,
                "topic_count": len(c.syllabus_items),
                "tests_completed": len(submitted_tests),
                "average_score": round(avg_score, 3),
                "mindmap_parsed_at": c.mindmap_parsed_at.isoformat() if c.mindmap_parsed_at else None,
                "created_at": c.created_at.isoformat(),
            })

        return classroom_list
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve classrooms: {exc}",
        )


@router.get(
    "/classrooms/{classroom_id}",
    summary="Get Classroom Details",
    description="Retrieves details and syllabus topics for a specific classroom with concept mastery.",
)
async def get_classroom_details(
    classroom_id: str,
    db_session: AsyncSession = Depends(get_db_session),
) -> Dict[str, Any]:
    try:
        stmt = (
            select(ClassroomEntity)
            .where(ClassroomEntity.id == classroom_id)
            .options(
                selectinload(ClassroomEntity.syllabus_items).selectinload(SyllabusItemEntity.concept)
            )
        )
        result = await db_session.execute(stmt)
        classroom = result.scalar_one_or_none()

        if not classroom:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Classroom with ID '{classroom_id}' not found.",
            )

        syllabus_list = []
        for item in classroom.syllabus_items:
            concept = item.concept
            syllabus_list.append({
                "id": item.id,
                "name": item.name,
                "description": item.description,
                "order_index": item.order_index,
                "concept_id": item.concept_id,
                "mastery_score": concept.mastery_score if concept else 0.0,
                "bloom_level": concept.bloom_level if concept else 1,
                "fsrs_stability": concept.fsrs_stability if concept else 1.0,
            })

        return {
            "id": classroom.id,
            "title": classroom.title,
            "description": classroom.description,
            "topic_query": classroom.topic_query,
            "mindmap_parsed_at": classroom.mindmap_parsed_at.isoformat() if classroom.mindmap_parsed_at else None,
            "created_at": classroom.created_at.isoformat(),
            "syllabus": syllabus_list,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve classroom details: {exc}",
        )


@router.patch(
    "/classrooms/{classroom_id}",
    summary="Update Classroom Metadata",
    description="Updates the title or description of a classroom.",
)
async def update_classroom(
    classroom_id: str,
    payload: UpdateClassroomSchema,
    db_session: AsyncSession = Depends(get_db_session),
) -> Dict[str, Any]:
    try:
        classroom = await db_session.get(ClassroomEntity, classroom_id)
        if not classroom:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Classroom with ID '{classroom_id}' not found.",
            )

        if payload.title is not None:
            classroom.title = payload.title.strip()
        if payload.description is not None:
            classroom.description = payload.description.strip()

        return {
            "id": classroom.id,
            "title": classroom.title,
            "description": classroom.description,
            "message": "Classroom updated successfully.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update classroom: {exc}",
        )


@router.delete(
    "/classrooms/{classroom_id}",
    summary="Delete Classroom",
    description="Permanently deletes a classroom and all associated syllabus topics, concepts, edges, test sessions, and mindmaps.",
)
async def delete_classroom(
    classroom_id: str,
    db_session: AsyncSession = Depends(get_db_session),
) -> Dict[str, str]:
    try:
        classroom = await db_session.get(ClassroomEntity, classroom_id)
        if not classroom:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Classroom with ID '{classroom_id}' not found.",
            )

        await db_session.delete(classroom)
        return {"message": f"Classroom '{classroom_id}' and all associated data have been permanently deleted."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete classroom: {exc}",
        )
