"""
===============================================================================
UNIT TESTS: MULTISOURCE CONTRAST INGESTION & VECTOR NORMALIZATION
===============================================================================
"""

import pytest
from app.services.multisource_contrast_ingestion_engine import contrast_engine


class TestTextSemanticChunking:
    """Test suite verifying text segmentation and boundary-preserving overlap."""

    def test_empty_or_whitespace_text_returns_empty_list(self):
        """Empty input or whitespace strings should produce no chunks."""
        assert contrast_engine._segment_text_semantic_chunks("") == []
        assert contrast_engine._segment_text_semantic_chunks("   \n\t  ") == []

    def test_short_text_single_chunk(self):
        """Text with fewer words than chunk_word_size should produce exactly one chunk."""
        raw_text = "Dynamic programming solves problems by combining solutions to subproblems."
        chunks = contrast_engine._segment_text_semantic_chunks(raw_text, chunk_word_size=50, overlap_word_size=10)
        assert len(chunks) == 1
        assert chunks[0] == raw_text

    def test_multi_chunk_segmentation_with_overlap(self):
        """Long text should be segmented into multiple chunks with correct overlapping words."""
        # Create 100 distinct words: "word_0 word_1 word_2 ... word_99"
        words = [f"word_{i}" for i in range(100)]
        raw_text = " ".join(words)

        # Chunk size: 40 words, Overlap: 10 words -> Step: 30 words
        # Chunk 0: words 0..39
        # Chunk 1: words 30..69
        # Chunk 2: words 60..99
        # Chunk 3: words 90..99
        chunks = contrast_engine._segment_text_semantic_chunks(raw_text, chunk_word_size=40, overlap_word_size=10)

        assert len(chunks) == 4
        # Verify first chunk
        assert chunks[0].startswith("word_0")
        assert chunks[0].endswith("word_39")
        # Verify second chunk starts with the overlap from first chunk
        assert chunks[1].startswith("word_30")
        assert chunks[1].endswith("word_69")


class TestVectorDimensionNormalization:
    """Test suite ensuring strict vector dimensionality conformance for embedded Qdrant."""

    def test_under_dimensioned_vector_is_zero_padded(self):
        """Vectors shorter than vector_dim (768) must be padded with zeros."""
        short_vec = [1.0, 2.0, 3.0]
        normalized = contrast_engine._normalize_vector_dimensions(short_vec)

        assert len(normalized) == contrast_engine.vector_dim
        assert normalized[:3] == [1.0, 2.0, 3.0]
        assert all(val == 0.0 for val in normalized[3:])

    def test_over_dimensioned_vector_is_truncated(self):
        """Vectors larger than vector_dim must be cleanly truncated to vector_dim."""
        target_dim = contrast_engine.vector_dim
        long_vec = [float(i) for i in range(target_dim + 100)]
        normalized = contrast_engine._normalize_vector_dimensions(long_vec)

        assert len(normalized) == target_dim
        assert normalized == long_vec[:target_dim]

    def test_exact_dimensioned_vector_preserved(self):
        """Vectors matching target vector_dim exactly must be preserved identically."""
        target_dim = contrast_engine.vector_dim
        exact_vec = [0.5] * target_dim
        normalized = contrast_engine._normalize_vector_dimensions(exact_vec)

        assert len(normalized) == target_dim
        assert normalized == exact_vec
