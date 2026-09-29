"""
===============================================================================
DYNAMIC SYLLABUS ROUTER & TOPOLOGICAL GRAPH API
===============================================================================

Exposes REST endpoints for:
- Exporting the canonical Grand Knowledge Graph for WebGL / React Flow visualization per classroom.
- Generating FSRS-driven concept recommendation queues for autonomous navigation.
- Ingesting new concept vertices and directed prerequisite dependencies scoped to classrooms.
"""

from __future__ import annotations
from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.supabase_persistence_manager import get_db_session
from app.services.knowledge_graph_engine import knowledge_graph_engine
from app.services.fsrs_spaced_retrieval_scheduler import fsrs_scheduler

router = APIRouter(prefix="/api/v1/syllabus", tags=["Dynamic Syllabus & Knowledge Graph"])


class CreateConceptRequestSchema(BaseModel):
    """Schema for registering a new concept vertex in the knowledge graph."""
    name: str = Field(..., min_length=1, max_length=255, description="Unique concept title")
    description: str = Field(default="", description="Detailed pedagogical summary of concept")
    classroom_id: Optional[str] = Field(default=None, description="Classroom UUID")
    position_x: Optional[float] = Field(default=None, description="Canvas X coordinate")
    position_y: Optional[float] = Field(default=None, description="Canvas Y coordinate")


class CreateKnowledgeEdgeRequestSchema(BaseModel):
    """Schema for creating a directed prerequisite dependency edge."""
    source_concept_id: str = Field(..., description="UUID of the prerequisite concept")
    target_concept_id: str = Field(..., description="UUID of the dependent concept")
    semantic_relation_label: str = Field(
        default="prerequisite_for", description="Semantic descriptor of relationship"
    )
    classroom_id: Optional[str] = Field(default=None, description="Classroom UUID")
    graph_partition: str = Field(
        default="grand", description="Target partition: 'grand' (canonical) or 'user' (mental model)"
    )


@router.get(
    "/graph",
    summary="Retrieve Grand Knowledge Graph",
    description="Returns all canonical concepts and prerequisite edges formatted for React Flow canvas rendering, optionally filtered by classroom.",
)
async def get_grand_knowledge_graph(
    classroom_id: Optional[str] = Query(default=None, description="Optional classroom UUID"),
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, list[dict[str, Any]]]:
    """Retrieve full React Flow node and edge definitions for the canonical Grand Schema."""
    return await knowledge_graph_engine.export_grand_graph_react_flow(
        db_session=db_session, classroom_id=classroom_id
    )


@router.get(
    "/route-next",
    summary="Compute FSRS Desirable Difficulty Recommendations",
    description="Ranks frontier concepts using FSRS retrievability decay and Bjork's desirable difficulty window (R ≈ 0.75) per classroom.",
)
async def get_recommended_study_queue(
    classroom_id: Optional[str] = Query(default=None, description="Optional classroom UUID"),
    top_k: int = Query(default=5, ge=1, le=20, description="Top K recommendations"),
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, list[dict[str, Any]]]:
    """Calculate the highest-leverage concepts ready for immediate study."""
    recommendations = await fsrs_scheduler.compute_priority_recommendations(
        db_session=db_session, classroom_id=classroom_id, top_k=top_k
    )
    return {"recommendations": recommendations}


@router.post(
    "/concepts",
    status_code=status.HTTP_201_CREATED,
    summary="Register Concept Node",
    description="Adds a new atomic concept node to the knowledge graph and persists it in Supabase.",
)
async def register_concept_node(
    payload: CreateConceptRequestSchema,
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Create a new concept vertex."""
    try:
        concept_entity = await knowledge_graph_engine.register_concept(
            db_session=db_session,
            name=payload.name.strip(),
            description=payload.description.strip(),
            classroom_id=payload.classroom_id,
            graph_partition="grand",
            position_x=payload.position_x,
            position_y=payload.position_y,
        )
        return {
            "id": concept_entity.id,
            "name": concept_entity.name,
            "classroom_id": concept_entity.classroom_id,
            "message": "Concept registered successfully.",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to register concept: {exc}",
        )


@router.post(
    "/edges",
    status_code=status.HTTP_201_CREATED,
    summary="Register Dependency Edge",
    description="Connects two concepts with a directed prerequisite dependency edge.",
)
async def register_dependency_edge(
    payload: CreateKnowledgeEdgeRequestSchema,
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Create a directed prerequisite edge."""
    try:
        edge_entity = await knowledge_graph_engine.register_knowledge_edge(
            db_session=db_session,
            source_concept_id=payload.source_concept_id,
            target_concept_id=payload.target_concept_id,
            semantic_relation_label=payload.semantic_relation_label,
            classroom_id=payload.classroom_id,
            graph_partition=payload.graph_partition,
        )
        return {
            "id": edge_entity.id,
            "source": edge_entity.source_concept_id,
            "target": edge_entity.target_concept_id,
            "classroom_id": edge_entity.classroom_id,
            "message": "Dependency edge registered successfully.",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to register edge: {exc}",
        )
