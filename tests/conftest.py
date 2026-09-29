"""
===============================================================================
SUPERLEARN BACKEND TEST FIXTURES & ISOLATION UTILITIES
===============================================================================

Provides shared fixtures for:
- Dependency-injected database session mocks
- Deterministic in-memory graph topologies
- Ephemeral CognitiveEngineSettings instances
- Asynchronous HTTP test clients for FastAPI endpoints
"""

from __future__ import annotations
from typing import AsyncGenerator
from unittest.mock import AsyncMock
import pytest
import networkx as nx
from httpx import AsyncClient, ASGITransport

from app.core.cognitive_config import CognitiveEngineSettings
from app.services.knowledge_graph_engine import knowledge_graph_engine, KnowledgeGraphEngine
from app.main import app
from app.db.supabase_persistence_manager import get_db_session


@pytest.fixture
def clean_settings() -> CognitiveEngineSettings:
    """Provide isolated configuration settings for testing."""
    return CognitiveEngineSettings(
        app_name="SuperLearn Cognitive Test Engine",
        app_env="development",
        cors_origins="http://localhost:3000",
        fsrs_target_retention=0.75,
        fsrs_stability_growth_factor=0.50,
        fsrs_difficulty_decay_factor=0.10,
        yerkes_dodson_optimal_seconds=120.0,
        yerkes_dodson_dispersion=45.0,
        vector_dimension=768,
    )


@pytest.fixture(autouse=True)
def reset_graph_topologies():
    """
    Ensure the global knowledge graph engine singleton is reset to clean 
    state before and after every test, preventing cross-test state leakage.
    """
    knowledge_graph_engine._grand_graph.clear()
    knowledge_graph_engine._user_mental_graph.clear()
    yield
    knowledge_graph_engine._grand_graph.clear()
    knowledge_graph_engine._user_mental_graph.clear()


@pytest.fixture
def populated_knowledge_graph() -> KnowledgeGraphEngine:
    """
    Populate knowledge_graph_engine with a canonical test topology:
    
    Structure:
      - Concept A (mastery=0.95): Root prerequisite, fully mastered.
      - Concept B (mastery=0.50): Dependent on A. Ready on frontier! (Prereq A >= 0.70, m(B) < 0.90)
      - Concept C (mastery=0.92): Already mastered. Excluded from frontier! (m(C) >= 0.90)
      - Concept D (mastery=0.10): Dependent on B. Blocked! (Prereq B is 0.50 < 0.70)
    """
    engine = knowledge_graph_engine
    g = engine._grand_graph

    # Add concepts
    g.add_node("concept-a", name="Concept A", mastery_score=0.95, fsrs_stability=10.0, fsrs_difficulty=3.0, last_retrieval_timestamp=None)
    g.add_node("concept-b", name="Concept B", mastery_score=0.50, fsrs_stability=2.0, fsrs_difficulty=5.0, last_retrieval_timestamp=None)
    g.add_node("concept-c", name="Concept C", mastery_score=0.92, fsrs_stability=15.0, fsrs_difficulty=2.0, last_retrieval_timestamp=None)
    g.add_node("concept-d", name="Concept D", mastery_score=0.10, fsrs_stability=1.0, fsrs_difficulty=6.0, last_retrieval_timestamp=None)

    # Add prerequisite edges: A -> B, B -> D
    g.add_edge("concept-a", "concept-b", id="edge-ab", semantic_relation_label="prerequisite_for")
    g.add_edge("concept-b", "concept-d", id="edge-bd", semantic_relation_label="prerequisite_for")

    return engine


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """Create a mock SQLAlchemy AsyncSession for testing database endpoints without live PostgreSQL."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.get = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
async def async_client(mock_db_session: AsyncMock) -> AsyncGenerator[AsyncClient, None]:
    """Provide an asynchronous test client bound to the FastAPI application with mocked DB dependencies."""
    app.dependency_overrides[get_db_session] = lambda: mock_db_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
    app.dependency_overrides.clear()
