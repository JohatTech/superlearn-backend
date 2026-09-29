"""
===============================================================================
MENTAL SCHEMA & CONFUSION COMPASS API ROUTER
===============================================================================

Exposes REST endpoints for:
- Managing the student's personal mental model schema graph per classroom.
- Uploading and parsing Mind Map images via Vision LLM into persistent DAG topologies.
- Persisting canvas layout coordinates (React Flow node positions).
- Executing the Confusion Compass graph diff against the canonical Grand Schema per classroom.
- Deleting and editing individual concept nodes.
"""

from __future__ import annotations
import logging
from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, status
from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.cognitive_config import cognitive_settings
from app.db.supabase_persistence_manager import get_db_session
from app.models.cognitive_domain_models import (
    ConceptEntity,
    KnowledgeEdgeEntity,
    ClassroomEntity,
    MindmapUploadEntity,
    get_current_utc_timestamp,
)
from app.services.knowledge_graph_engine import knowledge_graph_engine
from app.services.confusion_compass_diff_engine import confusion_compass_engine
from app.services.mindmap_extraction_service import mindmap_extraction_service

logger = logging.getLogger("superlearn.mental_schema_router")

router = APIRouter(prefix="/api/v1/schema", tags=["Mental Schema & Confusion Compass"])


class CreateUserConceptSchema(BaseModel):
    """Schema for adding a concept to the student's mental model."""
    name: str = Field(..., min_length=1, max_length=255, description="Concept name")
    description: str = Field(default="", description="Student's subjective understanding")
    classroom_id: Optional[str] = Field(default=None, description="Classroom UUID")
    position_x: Optional[float] = Field(default=None, description="Canvas X coordinate")
    position_y: Optional[float] = Field(default=None, description="Canvas Y coordinate")


class UpdateUserConceptSchema(BaseModel):
    """Schema for editing an existing concept bubble."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=255, description="New concept name")
    description: Optional[str] = Field(default=None, description="New description")


class CreateUserEdgeSchema(BaseModel):
    """Schema for adding a subjective dependency edge."""
    source_concept_id: str = Field(..., description="UUID of source concept")
    target_concept_id: str = Field(..., description="UUID of target concept")
    semantic_relation_label: str = Field(default="relates_to", description="Relationship label")
    classroom_id: Optional[str] = Field(default=None, description="Classroom UUID")


class NodePositionItem(BaseModel):
    id: str
    position: dict[str, float]


class SaveLayoutSchema(BaseModel):
    classroom_id: Optional[str] = None
    positions: List[NodePositionItem]


@router.get(
    "/user-graph",
    summary="Retrieve Student Mental Model Graph",
    description="Returns the user's personal concept graph formatted for React Flow canvas rendering, scoped to a classroom.",
)
async def get_user_mental_graph(
    classroom_id: Optional[str] = Query(default=None, description="Optional classroom UUID filter"),
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, list[dict[str, Any]]]:
    """Retrieve React Flow representation of student's current mental model."""
    return await knowledge_graph_engine.export_user_mental_graph_react_flow(
        db_session=db_session, classroom_id=classroom_id
    )


@router.post(
    "/user-concept",
    status_code=status.HTTP_201_CREATED,
    summary="Add Concept to Mental Schema",
    description="Inserts a concept into the student's mental model partition in Supabase.",
)
async def add_concept_to_mental_schema(
    payload: CreateUserConceptSchema,
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Add a concept to user schema."""
    try:
        concept = await knowledge_graph_engine.register_concept(
            db_session=db_session,
            name=payload.name.strip(),
            description=payload.description.strip(),
            classroom_id=payload.classroom_id,
            graph_partition="user",
            position_x=payload.position_x,
            position_y=payload.position_y,
        )
        return {
            "id": concept.id,
            "name": concept.name,
            "classroom_id": concept.classroom_id,
            "message": "Added to mental model schema.",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to add concept: {exc}",
        )


@router.patch(
    "/user-concept/{concept_id}",
    summary="Edit a Concept Bubble",
    description="Updates the name and/or description of an existing concept node in the user's mental model.",
)
async def update_user_concept(
    concept_id: str,
    payload: UpdateUserConceptSchema,
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Edit a concept bubble's label or description."""
    concept = await knowledge_graph_engine.update_concept(
        db_session=db_session,
        concept_id=concept_id,
        name=payload.name,
        description=payload.description,
    )
    if concept is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concept '{concept_id}' not found.",
        )
    return {
        "id": concept.id,
        "name": concept.name,
        "description": concept.description,
        "message": "Concept updated successfully.",
    }


@router.delete(
    "/user-concept/{concept_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a Concept Bubble",
    description="Permanently removes a concept node and all its connected edges from the user's mental model.",
)
async def delete_user_concept(
    concept_id: str,
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Delete a concept bubble and all edges connected to it."""
    deleted = await knowledge_graph_engine.delete_concept(
        db_session=db_session,
        concept_id=concept_id,
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concept '{concept_id}' not found.",
        )
    return {"deleted": True, "id": concept_id, "message": "Concept and its edges deleted."}


@router.post(
    "/user-edge",
    status_code=status.HTTP_201_CREATED,
    summary="Add Edge to Mental Schema",
    description="Inserts a subjective relationship edge into the student's mental model in Supabase.",
)
async def add_edge_to_mental_schema(
    payload: CreateUserEdgeSchema,
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Add an edge to user schema."""
    try:
        edge = await knowledge_graph_engine.register_knowledge_edge(
            db_session=db_session,
            source_concept_id=payload.source_concept_id,
            target_concept_id=payload.target_concept_id,
            semantic_relation_label=payload.semantic_relation_label,
            classroom_id=payload.classroom_id,
            graph_partition="user",
        )
        return {
            "id": edge.id,
            "source": edge.source_concept_id,
            "target": edge.target_concept_id,
            "classroom_id": edge.classroom_id,
            "message": "Edge added to mental model schema.",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to add edge: {exc}",
        )


@router.post(
    "/layout",
    summary="Persist Canvas Node Positions",
    description="Saves dragged React Flow visual positions (X, Y) to Supabase.",
)
async def save_canvas_layout(
    payload: SaveLayoutSchema,
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Persist node coordinate layout."""
    try:
        items = [{"id": item.id, "position": item.position} for item in payload.positions]
        updated_count = await knowledge_graph_engine.save_concept_positions(db_session, items)
        return {"updated_count": updated_count, "message": "Layout coordinates persisted."}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save layout: {exc}",
        )


@router.get(
    "/diff",
    summary="Execute Confusion Compass Graph Diff",
    description="Diffs the student's mental schema against the Grand Schema to detect misconceptions, inverted relationships, and missing links.",
)
async def execute_confusion_compass_diff(
    classroom_id: Optional[str] = Query(default=None, description="Optional classroom UUID"),
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Run structural graph diff to reveal misconceptions."""
    return await confusion_compass_engine.compute_graph_discrepancy_matrix(
        db_session=db_session, classroom_id=classroom_id
    )


@router.post(
    "/parse-mindmap",
    status_code=status.HTTP_200_OK,
    summary="Parse Mind Map Image & Persist DAG to Classroom (Full Replacement)",
    description=(
        "Ingests multipart image data, extracts concepts via OCR and arrows via Computer Vision, "
        "CLEARS the existing user mental model for the classroom, then persists fresh extracted data. "
        "Each upload produces a complete replacement — no stale concepts are retained."
    ),
)
async def parse_mindmap(
    file: UploadFile = File(..., description="Mind map image file (PNG, JPG, WEBP)"),
    classroom_id: Optional[str] = Form(default=None, description="Classroom UUID to attach extracted DAG"),
    db_session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """
    Extract DAG from uploaded mind map image without LLM relationship hallucination.
    Replaces the entire user graph — past concepts are cleared before ingesting new ones.
    """
    raw_content = await file.read()
    if not raw_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image file is empty.",
        )

    filename = file.filename or "mindmap.png"
    file_bytes = len(raw_content)
    size_kb = file_bytes / 1024.0

    logger.info(
        f"\n"
        f"╔══════════════════════════════════════════════════════════════════════════════╗\n"
        f"║            >>> INCOMING MIND MAP INGESTION REQUEST RECEIVED <<<              ║\n"
        f"╚══════════════════════════════════════════════════════════════════════════════╝\n"
        f"  • File Name    : {filename}\n"
        f"  • File Size    : {size_kb:.2f} KB ({file_bytes} bytes)\n"
        f"  • Content-Type : {file.content_type}\n"
        f"  • Classroom ID : {classroom_id or '(None - Global User Mental Model)'}\n"
        f"════════════════════════════════════════════════════════════════════════════════"
    )

    # 1. Deterministic Extraction Pipeline: OCR text boxes + CV arrow detection + Spatial builder
    extracted_data = await mindmap_extraction_service.extract_mindmap(raw_content)
    parsed_nodes: list[dict] = extracted_data.get("nodes", [])
    parsed_edges: list[dict] = extracted_data.get("edges", [])
    summary_text = extracted_data.get("summary", "Extracted concepts and connecting arrows from mind map.")
    metadata = extracted_data.get("metadata", {})

    logger.info(
        f"\n"
        f"╔══════════════════════════════════════════════════════════════════════════════╗\n"
        f"║          [STEP 6] DATABASE PERSISTENCE & REACT FLOW GRAPH EXPORT             ║\n"
        f"╚══════════════════════════════════════════════════════════════════════════════╝\n"
        f"  • Extraction Metadata : {metadata}\n"
        f"  • Extracted Concepts  : {len(parsed_nodes)} nodes\n"
        f"  • Extracted Edges     : {len(parsed_edges)} edges\n"
        f"────────────────────────────────────────────────────────────────────────────────"
    )

    # 2. CLEAR existing user-partition graph for this classroom before inserting new data.
    #    This ensures each upload fully replaces the previous mind map — no stale concepts.
    cleared_count = await knowledge_graph_engine.clear_user_graph(
        db_session=db_session,
        classroom_id=classroom_id,
    )
    logger.info(f"  • Stale User Graph Cleanup: Cleared {cleared_count} previous concepts for classroom '{classroom_id or 'Global'}'.")

    # 3. Persist Extracted Concepts into Supabase (user partition)
    created_concept_map: dict[str, ConceptEntity] = {}

    logger.info(f"  • Persisting {len(parsed_nodes)} concept bubbles into database:")
    for idx, node in enumerate(parsed_nodes):
        node_name = (node.get("label") or node.get("name") or f"Concept {idx+1}").strip()[:255]
        node_desc = node.get("description", "").strip()
        pos_data = node.get("position", {})
        pos_x = float(pos_data.get("x", (idx % 3) * 240 + 80))
        pos_y = float(pos_data.get("y", (idx // 3) * 150 + 80))

        concept_entity = await knowledge_graph_engine.register_concept(
            db_session=db_session,
            name=node_name,
            description=node_desc,
            classroom_id=classroom_id,
            graph_partition="user",
            position_x=pos_x,
            position_y=pos_y,
        )
        # Store by ID and by lowercased label for flexible edge linking
        created_concept_map[node.get("id", f"node_{idx+1}")] = concept_entity
        created_concept_map[node_name.lower()] = concept_entity

        logger.info(
            f"    [{idx+1:02d}] Concept: \"{node_name}\" -> DB ID: {concept_entity.id} "
            f"| Canvas Position: (X={pos_x:.1f}, Y={pos_y:.1f})"
        )

    # 4. Persist Extracted Edges into Supabase (user partition)
    created_edges_count = 0
    logger.info(f"  • Persisting {len(parsed_edges)} directed edges into database:")
    for idx, edge in enumerate(parsed_edges):
        src_key = edge.get("source", "")
        tgt_key = edge.get("target", "")
        src_name = edge.get("source_name", "").strip().lower()
        tgt_name = edge.get("target_name", "").strip().lower()
        rel_label = edge.get("relation", "relates_to").strip()[:255]

        # Look up by ID first, then by name
        src_entity = created_concept_map.get(src_key) or created_concept_map.get(src_name)
        tgt_entity = created_concept_map.get(tgt_key) or created_concept_map.get(tgt_name)

        if src_entity and tgt_entity and src_entity.id != tgt_entity.id:
            await knowledge_graph_engine.register_knowledge_edge(
                db_session=db_session,
                source_concept_id=src_entity.id,
                target_concept_id=tgt_entity.id,
                semantic_relation_label=rel_label,
                classroom_id=classroom_id,
                graph_partition="user",
            )
            created_edges_count += 1
            logger.info(
                f"    [{created_edges_count:02d}] Edge: \"{src_entity.name}\" ──[{rel_label}]──> \"{tgt_entity.name}\""
            )
        else:
            logger.warning(
                f"    [!] Edge skipped (unmatched endpoints): src_key='{src_key}', tgt_key='{tgt_key}', "
                f"src_found={bool(src_entity)}, tgt_found={bool(tgt_entity)}"
            )

    # 5. Record Mindmap Upload Entity in Supabase
    if classroom_id:
        mindmap_record = MindmapUploadEntity(
            classroom_id=classroom_id,
            filename=filename,
            parsed_nodes_count=len(parsed_nodes),
            parsed_edges_count=created_edges_count,
            extracted_summary=summary_text,
        )
        db_session.add(mindmap_record)

        classroom_record = await db_session.get(ClassroomEntity, classroom_id)
        if classroom_record:
            classroom_record.mindmap_parsed_at = get_current_utc_timestamp()

    await db_session.flush()

    # 6. Export Updated React Flow Graph for the user
    user_graph_export = await knowledge_graph_engine.export_user_mental_graph_react_flow(
        db_session=db_session, classroom_id=classroom_id
    )

    logger.info(
        f"\n"
        f"╔══════════════════════════════════════════════════════════════════════════════╗\n"
        f"║            >>> MIND MAP INGESTION PIPELINE COMPLETED <<<                     ║\n"
        f"╚══════════════════════════════════════════════════════════════════════════════╝\n"
        f"  • Output React Flow Nodes (Bubbles) : {len(user_graph_export['nodes'])}\n"
        f"  • Output React Flow Edges (Arrows)  : {len(user_graph_export['edges'])}\n"
        f"  • Summary                           : {summary_text}\n"
        f"════════════════════════════════════════════════════════════════════════════════\n"
    )

    return {
        "status": "success",
        "message": (
            f"Mind map replaced successfully: {len(parsed_nodes)} concepts, "
            f"{created_edges_count} directed edges. "
            f"Previous map cleared ({cleared_count} concepts removed)."
        ),
        "classroom_id": classroom_id,
        "summary": summary_text,
        "metadata": metadata,
        "nodes": user_graph_export["nodes"],
        "edges": user_graph_export["edges"],
    }
