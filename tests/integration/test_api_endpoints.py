"""
===============================================================================
INTEGRATION TESTS: FASTAPI HTTP ENDPOINTS & ROUTERS
===============================================================================
"""

import pytest
from httpx import AsyncClient
from app.services.knowledge_graph_engine import knowledge_graph_engine


class TestHealthCheckEndpoint:
    """Test suite for system diagnostics and health check endpoint."""

    async def test_health_check_returns_200_and_metadata(self, async_client: AsyncClient):
        """Verify GET /health returns service status and runtime metadata."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "healthy"
        assert "app_name" in payload
        assert "version" in payload
        assert "llm_provider" in payload
        assert "vector_store" in payload


class TestDynamicSyllabusEndpoints:
    """Test suite for /api/v1/syllabus endpoints."""

    async def test_get_grand_graph(self, async_client: AsyncClient, populated_knowledge_graph):
        """Verify GET /api/v1/syllabus/graph returns React Flow structure."""
        response = await async_client.get("/api/v1/syllabus/graph")
        assert response.status_code == 200
        payload = response.json()
        assert "nodes" in payload
        assert "edges" in payload
        assert len(payload["nodes"]) == 4

    async def test_get_recommended_study_queue(self, async_client: AsyncClient, populated_knowledge_graph):
        """Verify GET /api/v1/syllabus/route-next calculates FSRS recommendation queue."""
        response = await async_client.get("/api/v1/syllabus/route-next?top_k=3")
        assert response.status_code == 200
        payload = response.json()
        assert "recommendations" in payload
        assert isinstance(payload["recommendations"], list)


class TestMentalSchemaEndpoints:
    """Test suite for /api/v1/schema endpoints and Confusion Compass."""

    async def test_get_user_mental_graph(self, async_client: AsyncClient):
        """Verify GET /api/v1/schema/user-graph returns empty or existing user graph."""
        response = await async_client.get("/api/v1/schema/user-graph")
        assert response.status_code == 200
        payload = response.json()
        assert "nodes" in payload
        assert "edges" in payload

    async def test_execute_confusion_compass_diff(self, async_client: AsyncClient):
        """Verify GET /api/v1/schema/diff returns discrepancy matrix structure."""
        response = await async_client.get("/api/v1/schema/diff")
        assert response.status_code == 200
        payload = response.json()
        assert "missing_edges" in payload
        assert "false_positive_edges" in payload
        assert "inverted_edges" in payload
        assert "conflict_count" in payload
        assert isinstance(payload["conflict_count"], int)


class TestContrastIngestionEndpoints:
    """Test suite for /api/v1/contrast endpoints."""

    async def test_list_available_sources(self, async_client: AsyncClient):
        """Verify GET /api/v1/contrast/sources returns sources list."""
        response = await async_client.get("/api/v1/contrast/sources")
        assert response.status_code == 200
        payload = response.json()
        assert "sources" in payload
        assert isinstance(payload["sources"], list)

    async def test_align_contrast_missing_query_parameters(self, async_client: AsyncClient):
        """Verify GET /api/v1/contrast/align validates required query parameters."""
        # Missing all required parameters
        response = await async_client.get("/api/v1/contrast/align")
        assert response.status_code == 422

        # Missing query, only providing sources
        response = await async_client.get("/api/v1/contrast/align?source_a=docA&source_b=docB")
        assert response.status_code == 422

    async def test_align_contrast_empty_source_strings(self, async_client: AsyncClient):
        """Verify GET /api/v1/contrast/align rejects whitespace-only sources with 400 Bad Request."""
        response = await async_client.get("/api/v1/contrast/align?query=recursion&source_a=%20&source_b=%20")
        assert response.status_code == 400
        assert "Both 'source_a' and 'source_b'" in response.json()["detail"]


class TestBloomAssessmentEndpoints:
    """Test suite for /api/v1/test endpoints."""

    async def test_submit_answer_validation_errors(self, async_client: AsyncClient):
        """Verify POST /api/v1/test/submit rejects invalid or too-short answers."""
        # Short answer (< 5 characters) fails Pydantic schema validation (422)
        response = await async_client.post(
            "/api/v1/test/submit",
            json={"session_id": "test-session", "answer_text": "no", "effort_latency_seconds": 10},
        )
        assert response.status_code == 422

        # Blank whitespace answer fails 400 validation
        response = await async_client.post(
            "/api/v1/test/submit",
            json={"session_id": "test-session", "answer_text": "      ", "effort_latency_seconds": 10},
        )
        assert response.status_code == 400
        assert "Answer cannot be blank" in response.json()["detail"]
