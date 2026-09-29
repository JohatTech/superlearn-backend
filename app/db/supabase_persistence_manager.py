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
from sqlalchemy import text
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
    # Set fast 5-second socket timeout so firewall-blocked ports fail fast instead of freezing startup
    connect_args = {
        "prepared_statement_cache_size": 0,
        "statement_cache_size": 0,
        "timeout": 5.0,
        "command_timeout": 5.0,
    }
    # Enforce SSL for Supabase and remote cloud postgres databases
    if is_remote_postgres or "ssl=require" in db_url or "ssl=true" in db_url.lower():
        connect_args["ssl"] = "require"
else:
    # SQLite parameters
    connect_args = {"check_same_thread": False}

# Asynchronous Database Engine Instance configured for Supabase / PostgreSQL or local SQLite
engine_kwargs: dict[str, Any] = {
    "echo": (cognitive_settings.app_env == "development"),
    "connect_args": connect_args,
}
if is_postgres:
    engine_kwargs.update({
        "pool_pre_ping": True,
        "pool_size": cognitive_settings.db_pool_size,
        "max_overflow": cognitive_settings.db_max_overflow,
        "pool_timeout": cognitive_settings.db_pool_timeout,
    })

engine: AsyncEngine = create_async_engine(
    db_url,
    **engine_kwargs,
)

# Async Session Factory
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autoflush=False,
    expire_on_commit=False,
)


def switch_to_sqlite_fallback(reason: str = "") -> None:
    """
    Rebinds the SQLAlchemy engine and AsyncSessionLocal factory to local SQLite ('superlearn.db').
    Called automatically when Supabase PostgreSQL is unreachable due to firewall or network blocks.
    """
    global engine, AsyncSessionLocal, is_postgres, db_url
    if not is_postgres:
        return  # Already using SQLite
    
    logger.warning(
        f"Switching relational persistence engine to local SQLite fallback ('superlearn.db'). "
        f"Reason: {reason if reason else 'Manual override or network failure'}"
    )
    db_url = "sqlite+aiosqlite:///./superlearn.db"
    is_postgres = False
    
    engine = create_async_engine(
        db_url,
        echo=(cognitive_settings.app_env == "development"),
        connect_args={"check_same_thread": False},
    )
    AsyncSessionLocal = async_sessionmaker(
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
        Active SQLAlchemy AsyncSession bound to PostgreSQL or local SQLite fallback.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            logger.error(f"Database transaction error. Rolling back: {exc}")
            raise


async def _apply_auto_migrations(conn) -> None:
    """
    Applies automatic, idempotent column migrations for existing databases to ensure 
    schema parity with declared SQLAlchemy models (e.g. adding missing classroom_id).
    """
    try:
        if is_postgres:
            query = """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='cognitive_concepts' AND column_name='classroom_id') THEN
                    ALTER TABLE cognitive_concepts ADD COLUMN classroom_id VARCHAR(36) REFERENCES classrooms(id) ON DELETE CASCADE;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='cognitive_edges' AND column_name='classroom_id') THEN
                    ALTER TABLE cognitive_edges ADD COLUMN classroom_id VARCHAR(36) REFERENCES classrooms(id) ON DELETE CASCADE;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='adaptive_test_sessions' AND column_name='classroom_id') THEN
                    ALTER TABLE adaptive_test_sessions ADD COLUMN classroom_id VARCHAR(36) REFERENCES classrooms(id) ON DELETE CASCADE;
                END IF;
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='mindmap_uploads' AND column_name='classroom_id') THEN
                    ALTER TABLE mindmap_uploads ADD COLUMN classroom_id VARCHAR(36) REFERENCES classrooms(id) ON DELETE CASCADE;
                END IF;
            END $$;
            """
            await conn.execute(text(query))
        else:
            for table_name in ["cognitive_concepts", "cognitive_edges", "adaptive_test_sessions", "mindmap_uploads"]:
                res = await conn.execute(text(f"PRAGMA table_info({table_name})"))
                columns = [row[1] for row in res.fetchall()]
                if columns and "classroom_id" not in columns:
                    logger.info(f"Auto-migrating SQLite table '{table_name}': adding missing column 'classroom_id'")
                    await conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN classroom_id VARCHAR(36)"))
    except Exception as exc:
        logger.warning(f"Auto-migration check note: {exc}")


async def initialize_database_schema() -> None:
    """
    Bootstraps all declared relational tables in the target Supabase / PostgreSQL or local SQLite database.
    Performs a pre-flight connection test and automatically falls back to local SQLite if
    PostgreSQL is unreachable due to firewall or network restrictions.
    """
    global is_postgres
    logger.info(f"Initializing relational database schema (is_postgres={is_postgres}, url={db_url})...")
    
    if is_postgres:
        try:
            async with engine.begin() as conn:
                import app.models.cognitive_domain_models  # noqa: F401
                await conn.run_sync(Base.metadata.create_all)
                await _apply_auto_migrations(conn)
            logger.info("Supabase PostgreSQL connection verified and relational schema initialized.")
            return
        except Exception as exc:
            logger.warning(
                f"Supabase PostgreSQL host is unreachable ({exc}). "
                f"Automatically switching to local SQLite database fallback ('superlearn.db')."
            )
            switch_to_sqlite_fallback(reason=str(exc))

    # Initialize local SQLite schema
    async with engine.begin() as conn:
        import app.models.cognitive_domain_models  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)
        await _apply_auto_migrations(conn)
    logger.info("Local SQLite relational schema verified and initialized.")


def get_supabase_client():
    """
    Returns an initialized Sync Supabase Python Client operating over HTTPS (Port 443) HTTP REST API.
    Resolves URL and anon key from effective settings or environment variables.
    """
    url = cognitive_settings.effective_supabase_url
    key = cognitive_settings.effective_supabase_anon_key
    if url and key:
        try:
            from supabase import create_client
            return create_client(url, key)
        except Exception as exc:
            logger.warning(f"Could not initialize Supabase Sync HTTP Client: {exc}")
    return None


async def get_async_supabase_client():
    """
    Returns an initialized Async Supabase Python Client operating over HTTPS (Port 443) HTTP REST API.
    Resolves URL and anon key from effective settings or environment variables.
    """
    url = cognitive_settings.effective_supabase_url
    key = cognitive_settings.effective_supabase_anon_key
    if url and key:
        try:
            from supabase import create_async_client
            return await create_async_client(url, key)
        except Exception as exc:
            logger.warning(f"Could not initialize Supabase Async HTTP Client: {exc}")
    return None

