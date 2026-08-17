"""
===============================================================================
MENTAL SCHEMA & CONFUSION COMPASS API ROUTER
===============================================================================

Exposes REST endpoints for:
- Managing the student's personal mental model schema graph.
- Adding subjective concept nodes and dependency hypotheses.
- Executing the Confusion Compass graph diff against the canonical Grand Schema.
"""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.supabase_persistence_manager import get_db_session
from app.services.knowledge_graph_engine import knowledge_graph_engine
from app.services.confusion_compass_diff_engine import confusion_compass_engine

router = APIRouter(prefix="/api/v1/schema", tags=["Mental Schema & Confusion Compass"])


class CreateUserConceptSchema(BaseModel):
    """Schema for adding a concept to the student's mental model."""
    name: str = Field(..., min_length=1, max_length=255, description="Concept name")
    description: str = Field(default="", description="Student's subjective understanding")


class CreateUserEdgeSchema(BaseModel):
    """Schema for adding a subjective dependency edge."""
    source_concept_id: str = Field(..., description="UUID of source concept")
    target_concept_id: str = Field(..., description="UUID of target concept")
    semantic_relation_label: str = Field(default="relates_to", description="Relationship label")


@router.get(
    "/user-graph",
    summary="Retrieve Student Mental Model Graph",
    description="Returns the user's personal concept graph formatted for React Flow canvas rendering.",
)
async def get_user_mental_graph() -> dict[str, list[dict[str, Any]]]:
    """Retrieve React Flow representation of student's current mental model."""
    return knowledge_graph_engine.export_user_mental_graph_react_flow()


@router.post(
    "/user-concept",
    status_code=status.HTTP_201_CREATED,
    summary="Add Concept to Mental Schema",
    description="Inserts a concept into the student's mental model partition.",
)
async def add_concept_to_mental_schema(
    payload: CreateUserConceptSchema,
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, str]:
    """Add a concept to user schema."""
    try:
        concept = await knowledge_graph_engine.register_concept(
            db_session=db_session,
            name=payload.name.strip(),
            description=payload.description.strip(),
            graph_partition="user",
        )
        return {"id": concept.id, "name": concept.name, "message": "Added to mental model schema."}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to add concept: {exc}",
        )


@router.post(
    "/user-edge",
    status_code=status.HTTP_201_CREATED,
    summary="Add Edge to Mental Schema",
    description="Inserts a subjective relationship edge into the student's mental model.",
)
async def add_edge_to_mental_schema(
    payload: CreateUserEdgeSchema,
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, str]:
    """Add an edge to user schema."""
    try:
        edge = await knowledge_graph_engine.register_knowledge_edge(
            db_session=db_session,
            source_concept_id=payload.source_concept_id,
            target_concept_id=payload.target_concept_id,
            semantic_relation_label=payload.semantic_relation_label,
            graph_partition="user",
        )
        return {"id": edge.id, "message": "Edge added to mental model schema."}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to add edge: {exc}",
        )


@router.get(
    "/diff",
    summary="Execute Confusion Compass Graph Diff",
    description="Diffs the student's mental schema against the Grand Schema to detect misconceptions, inverted relationships, and missing links.",
)
async def execute_confusion_compass_diff() -> dict[str, Any]:
    """Run structural graph diff to reveal misconceptions."""
    return confusion_compass_engine.compute_graph_discrepancy_matrix()
