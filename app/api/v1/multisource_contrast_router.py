"""
===============================================================================
MULTISOURCE CONTRAST INGESTION & SEARCH API ROUTER
===============================================================================

Exposes REST endpoints for:
- Uploading and indexing plain text or PDF documents into embedded Qdrant (pure Python).
- Aligning dual-source passages side-by-side for comparative contrast reading.
- Performing semantic similarity searches across all indexed documents.
"""

from __future__ import annotations
import io
import logging
from typing import Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, status
from pypdf import PdfReader
from app.services.multisource_contrast_ingestion_engine import contrast_engine

logger = logging.getLogger("superlearn.contrast_router")

router = APIRouter(prefix="/api/v1", tags=["Multisource Contrast & Semantic Ingestion"])


@router.post(
    "/ingest/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload & Index Document",
    description="Extracts text from PDF/TXT, segments into 15% overlapping chunks, embeds via Ollama/Azure, and indices into embedded Qdrant.",
)
async def upload_and_ingest_document(
    file: UploadFile = File(..., description="Document file (PDF, TXT, MD)"),
    source_name: str = Form(..., description="Canonical citation title for the document"),
) -> dict[str, Any]:
    """Upload and vectorize a document into local embedded Qdrant storage."""
    raw_content = await file.read()
    content_type = file.content_type or ""
    filename = file.filename or ""

    if "pdf" in content_type or filename.lower().endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(raw_content))
            extracted_pages = [page.extract_text() or "" for page in reader.pages]
            parsed_text = "\n".join(extracted_pages).strip()
        except Exception as exc:
            logger.error(f"PDF extraction error: {exc}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse PDF document: {exc}",
            )
    else:
        parsed_text = raw_content.decode("utf-8", errors="replace").strip()

    if not parsed_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded document contains no readable text.",
        )

    chunks_stored = await contrast_engine.ingest_document_corpus(
        document_text=parsed_text,
        source_identifier=source_name.strip(),
    )

    return {
        "source_name": source_name.strip(),
        "chunks_stored": chunks_stored,
        "status": "indexed_successfully",
    }


@router.get(
    "/contrast/align",
    summary="Align Cross-Document Contrast Passages",
    description="Retrieves the most semantically relevant excerpts from two different documents for side-by-side comparison.",
)
async def align_cross_document_contrast(
    query: str = Query(..., description="Concept topic or search query"),
    source_a: str = Query(..., description="Source document A identifier"),
    source_b: str = Query(..., description="Source document B identifier"),
    top_k: int = Query(default=3, ge=1, le=10, description="Passages per document"),
) -> dict[str, Any]:
    """Align corresponding passages between two sources for side-by-side study."""
    if not source_a.strip() or not source_b.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both 'source_a' and 'source_b' query parameters must be specified.",
        )

    contrast_payload = await contrast_engine.align_cross_document_contrast(
        concept_query=query.strip(),
        source_name_a=source_a.strip(),
        source_name_b=source_b.strip(),
        top_k=top_k,
    )
    return contrast_payload


@router.get(
    "/contrast/search",
    summary="Semantic Corpus Search",
    description="Executes cosine similarity search across all indexed document chunks in embedded Qdrant.",
)
async def search_corpus_passages(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20, description="Max results"),
    source: str | None = Query(default=None, description="Optional document source filter"),
) -> dict[str, list[dict[str, Any]]]:
    """Search for relevant passages across all indexed materials."""
    results = await contrast_engine.search_semantic_passages(
        query=query.strip(), top_k=top_k, source_filter=source
    )
    return {"results": results}


@router.get(
    "/contrast/sources",
    summary="List Ingested Document Sources",
    description="Returns all unique document names currently stored in the embedded Qdrant index.",
)
async def list_available_sources() -> dict[str, list[str]]:
    """List all indexed document sources."""
    sources = contrast_engine.list_ingested_sources()
    return {"sources": sources}
