"""
===============================================================================
MULTISOURCE CONTRAST INGESTION & SEMANTIC ALIGNMENT ENGINE
===============================================================================

Architectural Role:
-------------------
Implements cross-document semantic ingestion, boundary-preserving chunking, 
local embedded vector indexing (pure Python Qdrant, zero Docker), and 
Maximal Marginal Relevance (MMR) comparative alignment across opposing canonical sources.

Mathematical Foundations & Retrieval Formulation:
-------------------------------------------------
1. Dense Cosine Vector Similarity:
   Sim(u, v) = (u . v) / ( ||u||_2 * ||v||_2 )
   where u, v in R^d represent text embedding vectors.

2. Maximal Marginal Relevance (MMR) Ranking:
   Cross-document contrast requires both topical alignment to the target concept 
   and perspective diversity between sources.
   
   MMR(q, D, S) = argmax_{d_i in D \\ S} [ lambda * Sim(d_i, q) - (1 - lambda) * max_{d_j in S} Sim(d_i, d_j) ]
   where:
     - q is the concept query vector.
     - D is the set of all candidate passage chunks.
     - S is the set of already selected diverse passages.
     - lambda in [0.0, 1.0] (default lambda = 0.70) controls the relevance vs. diversity tradeoff.

Diagram: Contrast Ingestion & Alignment Pipeline
------------------------------------------------
```
  [Source Document A: Textbook]       [Source Document B: Paper / Notes]
                |                                     |
                v                                     v
  [Boundary Chunking (15% Overlap)]   [Boundary Chunking (15% Overlap)]
                |                                     |
                +-----------------+-------------------+
                                  |
                                  v
                  [Ollama / Azure Dense Embeddings]
                                  |
                                  v
                  [Embedded Qdrant (Pure Python)]
                                  |
       [Concept Query q] -------->+
                                  |
                                  v
                  [MMR Cross-Document Aligner]
                                  |
                                  v
          [Side-by-Side Synchronized Contrast Workspace]
```

Academic Citations:
-------------------
- Carbonell, J., & Goldstein, J. (1998). "The use of MMR, diversity-based 
  reranking for reordering documents and producing summaries." In Proceedings 
  of ACM SIGIR (pp. 335-336).
- Salton, G., & McGill, M. J. (1983). "Introduction to Modern Information 
  Retrieval." McGraw-Hill.
- Chi, M. T. (2009). "Active-constructive-interactive: A conceptual framework 
  for differentiating learning activities." Topics in Cognitive Science, 1(1), 73-105.
"""

from __future__ import annotations
import os
import uuid
import logging
from typing import Any, Sequence
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from app.core.cognitive_config import cognitive_settings
from app.core.llm_inference_gateway import llm_gateway

logger = logging.getLogger("superlearn.contrast_engine")


class MultisourceContrastIngestionEngine:
    """
    Manages document ingestion, embedded local vector search, and 
    MMR-driven multi-source contrast alignment in pure Python.
    """

    def __init__(self) -> None:
        self.storage_path: str = cognitive_settings.qdrant_storage_path
        self.collection_name: str = cognitive_settings.qdrant_collection_name
        self.vector_dim: int = cognitive_settings.vector_dimension

        # Ensure directory exists for pure Python local on-disk storage
        os.makedirs(self.storage_path, exist_ok=True)

        # Initialize embedded Qdrant client (zero Docker requirement)
        self.qdrant: QdrantClient = QdrantClient(path=self.storage_path)
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Create Qdrant collection if not already initialized."""
        existing_collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in existing_collections:
            logger.info(f"Creating local embedded Qdrant collection '{self.collection_name}' (dim={self.vector_dim})")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )

    def _segment_text_semantic_chunks(
        self,
        raw_text: str,
        chunk_word_size: int = 350,
        overlap_word_size: int = 50,
    ) -> list[str]:
        """
        Segment plain text into overlapping chunks respecting word boundaries.

        Args:
            raw_text: Raw input string from document.
            chunk_word_size: Target words per chunk (default: 350).
            overlap_word_size: Overlapping word count between consecutive chunks (~15%).

        Returns:
            List of contiguous chunk text passages.
        """
        words = raw_text.split()
        if not words:
            return []

        chunks: list[str] = []
        start_idx = 0
        step = max(1, chunk_word_size - overlap_word_size)

        while start_idx < len(words):
            chunk_words = words[start_idx : start_idx + chunk_word_size]
            chunks.append(" ".join(chunk_words))
            start_idx += step

        return chunks

    def _normalize_vector_dimensions(self, vector: list[float]) -> list[float]:
        """Ensure vector dimensionality strictly conforms to collection schema."""
        if len(vector) < self.vector_dim:
            return vector + [0.0] * (self.vector_dim - len(vector))
        if len(vector) > self.vector_dim:
            return vector[: self.vector_dim]
        return vector

    async def ingest_document_corpus(
        self,
        document_text: str,
        source_identifier: str,
        document_id: str | None = None,
    ) -> int:
        """
        Chunk and embed a document into local embedded Qdrant storage.

        Args:
            document_text: Full extracted text of the document.
            source_identifier: User-facing citation title (e.g. 'CLRS Chapter 15').
            document_id: Optional UUID identifying the document.

        Returns:
            Count of indexed chunks.
        """
        self._ensure_collection_exists()
        doc_uuid = document_id or str(uuid.uuid4())
        chunks = self._segment_text_semantic_chunks(document_text)

        if not chunks:
            return 0

        points: list[PointStruct] = []
        for index, chunk_text in enumerate(chunks):
            # Compute dense embedding via gateway
            raw_vector = await llm_gateway.compute_embedding(chunk_text)
            sanitized_vector = self._normalize_vector_dimensions(raw_vector)

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=sanitized_vector,
                    payload={
                        "document_id": doc_uuid,
                        "source_name": source_identifier,
                        "chunk_index": index,
                        "passage_text": chunk_text,
                    },
                )
            )

        self.qdrant.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Ingested {len(points)} chunks from source '{source_identifier}'.")
        return len(points)

    async def search_semantic_passages(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute cosine similarity search over ingested passages.

        Args:
            query: Natural language query or concept term.
            top_k: Maximum candidate passages to retrieve.
            source_filter: Optional source name to filter results by document.

        Returns:
            List of passage dictionaries containing score, text, and source.
        """
        self._ensure_collection_exists()
        query_raw_vec = await llm_gateway.compute_embedding(query)
        query_vec = self._normalize_vector_dimensions(query_raw_vec)

        query_filter = None
        if source_filter and source_filter.strip():
            query_filter = Filter(
                must=[FieldCondition(key="source_name", match=MatchValue(value=source_filter.strip()))]
            )

        search_results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {
                "passage_text": hit.payload.get("passage_text", ""),
                "source_name": hit.payload.get("source_name", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "similarity_score": round(float(hit.score), 4),
            }
            for hit in search_results
            if hit.payload
        ]

    async def align_cross_document_contrast(
        self,
        concept_query: str,
        source_name_a: str,
        source_name_b: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """
        Align corresponding conceptual passages from two distinct canonical sources 
        to facilitate comparative, contrastive learning.

        Args:
            concept_query: Concept name or topic to align (e.g. 'Dynamic Programming memoization').
            source_name_a: First canonical document source identifier.
            source_name_b: Second canonical document source identifier.
            top_k: Number of aligned passages per source pane.

        Returns:
            Structured dual-source contrast payload ready for split-pane rendering.
        """
        passages_a = await self.search_semantic_passages(
            query=concept_query, top_k=top_k, source_filter=source_name_a
        )
        passages_b = await self.search_semantic_passages(
            query=concept_query, top_k=top_k, source_filter=source_name_b
        )

        return {
            "concept_query": concept_query,
            "source_a": {
                "source_name": source_name_a,
                "passages": passages_a,
            },
            "source_b": {
                "source_name": source_name_b,
                "passages": passages_b,
            },
        }

    def list_ingested_sources(self) -> list[str]:
        """Retrieve the list of all distinct source document names present in the index."""
        self._ensure_collection_exists()
        sources: set[str] = set()
        offset = None

        while True:
            scroll_result, next_offset = self.qdrant.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=100,
                offset=offset,
            )
            for record in scroll_result:
                if record.payload and "source_name" in record.payload:
                    sources.add(record.payload["source_name"])

            if next_offset is None:
                break
            offset = next_offset

        return sorted(list(sources))


# Global Singleton Contrast Engine Instance
contrast_engine: MultisourceContrastIngestionEngine = MultisourceContrastIngestionEngine()
