"""
===============================================================================
SUPERLEARN COGNITIVE LEARNING ENGINE — MAIN APPLICATION SERVER
===============================================================================

Core Application Lifecycle & HTTP Gateway:
- Initializes Supabase relational schema on startup.
- Synchronizes in-memory NetworkX knowledge graph topologies.
- Embeds local Qdrant vector database (Pure Python, Zero Docker).
- Mounts REST routers for Dynamic Syllabus, Contrast Reader, Schema & Testing.
"""

# Relational persistence trigger reload for config update
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.cognitive_config import cognitive_settings
from app.db.supabase_persistence_manager import (
    initialize_database_schema,
    AsyncSessionLocal,
)
from app.services.knowledge_graph_engine import knowledge_graph_engine
from app.api.v1 import (
    dynamic_syllabus_router,
    multisource_contrast_router,
    mental_schema_router,
    bloom_assessment_router,
    syllabus_master_router,
    study_materials_router,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO if cognitive_settings.app_env != "development" else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("superlearn.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application Lifespan Event Handler.
    Initializes database schema and synchronizes in-memory graph topologies upon startup.
    """
    logger.info(f"Starting {cognitive_settings.app_name} v{cognitive_settings.app_version}...")

    # 1. Initialize Supabase Relational Schema
    try:
        await initialize_database_schema()
        logger.info("Database schema initialized.")
    except Exception as exc:
        logger.warning(
            f"Relational schema check deferred (Supabase connection may require configured credentials): {exc}"
        )

    # 2. Synchronize In-Memory Knowledge Graph from Database
    try:
        async with AsyncSessionLocal() as session:
            await knowledge_graph_engine.synchronize_from_database(session)
        logger.info("Knowledge graph synchronized into memory.")
    except Exception as exc:
        logger.warning(f"In-memory graph hydration skipped (empty DB or offline): {exc}")

    yield

    logger.info("Shutting down SuperLearn Cognitive Engine...")


# FastAPI Application Instance
app = FastAPI(
    title=cognitive_settings.app_name,
    description=(
        "Autonomous closed-loop cognitive learning engine powered by FSRS spaced repetition, "
        "Bloom's Taxonomy assessments, multi-source contrast ingestion, and Confusion Compass graph diffing."
    ),
    version=cognitive_settings.app_version,
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS) for Frontend Client
app.add_middleware(
    CORSMiddleware,
    allow_origins=cognitive_settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Domain Routers
app.include_router(dynamic_syllabus_router)
app.include_router(multisource_contrast_router)
app.include_router(mental_schema_router)
app.include_router(bloom_assessment_router)
app.include_router(syllabus_master_router)
app.include_router(study_materials_router)



@app.get(
    "/health",
    summary="Service Health Status",
    description="Returns runtime status of cognitive engine, active LLM provider, and embedded vector store.",
    tags=["System Diagnostics"],
)
async def health_check() -> dict[str, str]:
    """Health check endpoint for system monitoring."""
    return {
        "status": "healthy",
        "app_name": cognitive_settings.app_name,
        "version": cognitive_settings.app_version,
        "llm_provider": cognitive_settings.llm_provider,
        "llm_model": cognitive_settings.ollama_model if cognitive_settings.llm_provider == "ollama" else cognitive_settings.azure_openai_deployment,
        "vector_store": "embedded_qdrant_pure_python",
    }
