"""
===============================================================================
SQLITE TO SUPABASE POSTGRESQL DATA MIGRATION UTILITY
===============================================================================

Architectural Role:
-------------------
Migrates offline study data (classrooms, syllabus items, concept nodes, knowledge edges,
test session logs, mindmap uploads, and cognitive profiles) from local SQLite ('superlearn.db')
into Supabase PostgreSQL.

Usage:
------
1. Ensure your network allows outbound connections to Supabase (Ports 5432 / 6543).
2. Configure DATABASE_URL or SUPERLEARN_POSTGRES_URL in .env.
3. Run from the backend directory:
   python scripts/migrate_sqlite_to_supabase.py
"""

from __future__ import annotations
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add backend directory to sys.path to resolve app imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.core.cognitive_config import cognitive_settings
from app.models.cognitive_domain_models import (
    AdaptiveTestSessionEntity,
    Base,
    ClassroomEntity,
    ConceptEntity,
    KnowledgeEdgeEntity,
    LearnerCognitiveProfileEntity,
    MindmapUploadEntity,
    SyllabusItemEntity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate_sqlite_to_supabase")


async def test_postgres_connection(pg_engine) -> bool:
    """Verifies that the Supabase PostgreSQL connection is reachable."""
    try:
        async with pg_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error(f"Failed to connect to Supabase PostgreSQL: {exc}")
        return False


async def migrate_data() -> None:
    """
    Idempotently migrates all relational entities from local SQLite to Supabase PostgreSQL.
    """
    sqlite_url = "sqlite+aiosqlite:///./superlearn.db"

    # Get raw postgres URL directly from settings/env to bypass USE_LOCAL_SQLITE override
    raw_pg_url = (
        cognitive_settings.database_url
        or cognitive_settings.superlearn_postgres_url_non_pooling
        or cognitive_settings.superlearn_postgres_url
        or cognitive_settings.superlearn_postgres_prisma_url
        or cognitive_settings.postgres_url_non_pooling
        or cognitive_settings.postgres_url
        or os.environ.get("DATABASE_URL")
        or os.environ.get("SUPERLEARN_POSTGRES_URL_NON_POOLING")
        or os.environ.get("SUPERLEARN_POSTGRES_URL")
        or ""
    ).strip()

    if not raw_pg_url or not any(proto in raw_pg_url.lower() for proto in ["postgres://", "postgresql://"]):
        logger.error(
            "No valid PostgreSQL/Supabase database URL found in configuration or environment."
        )
        logger.error(
            "Please configure SUPERLEARN_POSTGRES_URL or DATABASE_URL in your .env file."
        )
        sys.exit(1)

    # Format URL for asyncpg dialect
    pg_url = raw_pg_url
    if "://" in pg_url:
        scheme, rest = pg_url.split("://", 1)
        if "?" in rest:
            base_part, query_part = rest.split("?", 1)
            clean_params = [
                p
                for p in query_part.split("&")
                if not any(
                    bad in p.lower()
                    for bad in ["pgbouncer", "supa", "connection_limit"]
                )
            ]
            query_str = "&".join(clean_params)
            rest = f"{base_part}?{query_str}" if query_str else base_part
        pg_url = f"postgresql+asyncpg://{rest}"
    else:
        pg_url = f"postgresql+asyncpg://{pg_url}"

    if "sslmode=require" in pg_url:
        pg_url = pg_url.replace("sslmode=require", "ssl=require")

    logger.info("Connecting to local SQLite database ('superlearn.db')...")
    sqlite_engine = create_async_engine(
        sqlite_url, connect_args={"check_same_thread": False}
    )
    sqlite_session_factory = async_sessionmaker(
        sqlite_engine, expire_on_commit=False
    )

    logger.info("Testing connection to Supabase PostgreSQL...")
    pg_connect_args = {
        "prepared_statement_cache_size": 0,
        "statement_cache_size": 0,
        "timeout": 10.0,
        "command_timeout": 10.0,
        "ssl": "require",
    }
    pg_engine = create_async_engine(pg_url, connect_args=pg_connect_args)

    if not await test_postgres_connection(pg_engine):
        logger.error(
            "Cannot proceed with migration. Supabase PostgreSQL host is unreachable."
        )
        logger.error(
            "Ensure your network firewall permits outbound TCP traffic on ports 5432 / 6543."
        )
        await sqlite_engine.dispose()
        await pg_engine.dispose()
        sys.exit(1)

    pg_session_factory = async_sessionmaker(pg_engine, expire_on_commit=False)

    # Ensure schema exists on PostgreSQL
    async with pg_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Starting data migration from local SQLite to Supabase PostgreSQL...")

    entity_classes = [
        ("Classrooms", ClassroomEntity),
        ("Cognitive Concepts", ConceptEntity),
        ("Syllabus Items", SyllabusItemEntity),
        ("Knowledge Edges", KnowledgeEdgeEntity),
        ("Adaptive Test Sessions", AdaptiveTestSessionEntity),
        ("Mindmap Uploads", MindmapUploadEntity),
        ("Learner Profiles", LearnerCognitiveProfileEntity),
    ]

    total_migrated = 0

    async with sqlite_session_factory() as sqlite_session, pg_session_factory() as pg_session:
        for label, model_cls in entity_classes:
            result = await sqlite_session.execute(select(model_cls))
            records = result.scalars().all()

            if not records:
                logger.info(f"[{label}] No local SQLite records found.")
                continue

            count = 0
            for record in records:
                # Merge into PG session (idempotent upsert by primary key)
                await pg_session.merge(record)
                count += 1

            await pg_session.commit()
            logger.info(
                f"[{label}] Successfully migrated/upserted {count} records into Supabase PostgreSQL."
            )
            total_migrated += count

    logger.info(
        f"Migration complete! Total entities synchronized to Supabase: {total_migrated}"
    )

    await sqlite_engine.dispose()
    await pg_engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate_data())
