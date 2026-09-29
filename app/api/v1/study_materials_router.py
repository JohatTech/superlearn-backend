"""
===============================================================================
STUDY MATERIALS & REFERENCES SEARCH API ROUTER
===============================================================================

Exposes REST endpoints for searching, reading, contrasting, and streaming
study references (books, papers, web articles, open courseware) per syllabus concept.
Powered by StudyMaterialsAgent and LinkVerifier.
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Query, HTTPException, status
from fastapi.responses import StreamingResponse
from app.agents.study_materials_agent import study_materials_agent

logger = logging.getLogger("superlearn.study_materials_router")

router = APIRouter(prefix="/api/v1", tags=["Study Materials & References Search"])


@router.get(
    "/references/search",
    summary="Search References per Syllabus Concept",
    description="Invokes StudyMaterialsAgent to search, read, contrast, and decide best suitable study references for a syllabus concept.",
)
async def search_references(
    concept_name: str = Query(..., description="Name of syllabus concept to search references for"),
    concept_description: Optional[str] = Query(default="", description="Optional concept context or description"),
) -> Dict[str, Any]:
    """Search and contrast study materials for a specific syllabus concept (returns full set)."""
    if not concept_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="concept_name parameter is required and cannot be empty.",
        )

    try:
        results = await study_materials_agent.recommend_study_materials(
            concept_name=concept_name.strip(),
            concept_description=concept_description or "",
        )
        return results
    except Exception as exc:
        logger.error(f"Error in search_references endpoint: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve study references: {exc}",
        )


@router.get(
    "/references/stream",
    summary="Stream Verified References per Syllabus Concept (Flush in Real-Time)",
    description="Streams verified active reference hyperlinks progressively one-by-one as they are discovered and double-checked by LinkVerifier.",
)
async def stream_references(
    concept_name: str = Query(..., description="Name of syllabus concept to search references for"),
    concept_description: Optional[str] = Query(default="", description="Optional concept context or description"),
) -> StreamingResponse:
    """Stream verified active references line-by-line as NDJSON as soon as each item is double-checked."""
    if not concept_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="concept_name parameter is required and cannot be empty.",
        )

    async def event_generator():
        try:
            async for ref in study_materials_agent.stream_study_references(
                concept_name=concept_name.strip(),
                concept_description=concept_description or "",
            ):
                yield json.dumps(ref) + "\n"
        except Exception as exc:
            logger.error(f"Error streaming references for '{concept_name}': {exc}")
            err_payload = {"error": str(exc), "concept_name": concept_name}
            yield json.dumps(err_payload) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable proxy buffering to ensure immediate flush
        }
    )
