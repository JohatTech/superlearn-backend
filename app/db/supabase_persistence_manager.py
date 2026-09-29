"""
===============================================================================
SUPABASE RELATIONAL PERSISTENCE & CONNECTION LIFECYCLE MANAGER
===============================================================================

Architectural Role:
-------------------
Manages asynchronous database connection pooling, transactional boundaries, 
and ORM session lifecycles for PostgreSQL hosted on Supabase (or local SQLite fallback).

Persistence Guarantees:
-----------------------
- ACID transaction isolation for classrooms, concept mastery states, test sessions, and mindmaps.
- High-efficiency connection pooling via SQLAlchemy `asyncpg` dialect.
- Graceful connection recycling with automated ping verification (`pool_pre_ping=True`).
- PgBouncer / Supavisor transaction pooler compatibility (prepared_statement_cache_size=0).
- Automatic TLS/SSL certificate negotiation for Supabase cloud hosts.
"""

from __future__ import annotations
import logging
import ssl
from typing import AsyncGenerator, Any
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from app.core.cognitive_config import cognitive_settings

logger = logging.getLogger("superlearn.persistence")


class Base(DeclarativeBase):
    """
    SQLAlchemy Declarative Base class serving as the metadata registry 
    for all mapped cognitive and pedagogical entities.
    """
    pass


# Determine connection parameters depending on dialect
db_url = cognitive_settings.async_database_url
is_postgres = "postgresql" in db_url
is_remote_postgres = is_postgres and any(
    domain in db_url.lower() for domain in ["supabase", "render.com", "neon.tech", "pooler", "amazonaws.com", "azure"]
)

connect_args: dict[str, Any] = {}
if is_postgres:
    # Disable prepared statement caching for Supabase transaction pooler (PgBouncer/Supavisor)
    connect_args = {
        "prepared_statement_cache_size": 0,
        "statement_cache_size": 0,
    }
    # Enforce SSL for Supabase and remote cloud postgres databases
    if is_remote_postgres or "ssl=require" in db_url or "ssl=true" in db_url.lower():
        connect_args["ssl"] = "require"
else:
    # SQLite parameters
    connect_args = {"check_same_thread": False}

# Asynchronous Database Engine Instance configured for Supabase / PostgreSQL
engine: AsyncEngine = create_async_engine(
    db_url,
    echo=(cognitive_settings.app_env == "development"),
    pool_pre_ping=True,
    pool_size=cognitive_settings.db_pool_size if is_postgres else 5,
    max_overflow=cognitive_settings.db_max_overflow if is_postgres else 10,
    pool_timeout=cognitive_settings.db_pool_timeout,
    connect_args=connect_args,
)

# Async Session Factory
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autoflush=False,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an isolated asynchronous database session per request.
    Enforces automatic rollback upon unhandled exceptions and commits upon successful completion.

    Yields:
        Active SQLAlchemy AsyncSession bound to Supabase PostgreSQL.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            logger.error(f"Database transaction error. Rolling back: {exc}")
            raise


async def initialize_database_schema() -> None:
    """
    Bootstraps all declared relational tables in the target Supabase / PostgreSQL database.
    Called automatically during application lifespan startup event.
    """
    logger.info(f"Initializing relational database schema (is_postgres={is_postgres})...")
    async with engine.begin() as conn:
        # Import models inside function to ensure complete metadata registration
        import app.models.cognitive_domain_models  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Relational schema verified and initialized.")


def get_supabase_client():
    """
    Returns an initialized Supabase Python Client if URL and anon key are configured.
    Returns None if credentials are not provided.
    """
    if cognitive_settings.supabase_url and cognitive_settings.supabase_anon_key:
        try:
            from supabase import create_client, Client
            return create_client(
                cognitive_settings.supabase_url,
                cognitive_settings.supabase_anon_key,
            )
        except Exception as exc:
            logger.warning(f"Could not initialize Supabase Python SDK client: {exc}")
    return None
