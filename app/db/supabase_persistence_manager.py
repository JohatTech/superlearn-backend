"""
===============================================================================
SUPABASE RELATIONAL PERSISTENCE & CONNECTION LIFECYCLE MANAGER
===============================================================================

Architectural Role:
-------------------
Manages asynchronous database connection pooling, transactional boundaries, 
and ORM session lifecycles for PostgreSQL hosted on Supabase.

Persistence Guarantees:
-----------------------
- ACID transaction isolation for user test submissions and concept mastery states.
- High-efficiency connection pooling via SQLAlchemy `asyncpg` dialect.
- Graceful connection recycling with automated ping verification (`pool_pre_ping=True`).

Diagram: Data Persistence Tier
------------------------------
```
+-----------------------------------+
|     FastAPI Application Layer     |
+-----------------+-----------------+
                  | (Async Dependency Injection)
                  v
+-----------------+-----------------+
|   AsyncSession Context Manager    |
| (Transaction Commit / Rollback)   |
+-----------------+-----------------+
                  |
                  v
+-----------------+-----------------+
|     SQLAlchemy Engine Pool        |
| (asyncpg async connection pool)   |
+-----------------+-----------------+
                  | (TLS 1.3 / TCP:5432)
                  v
+-----------------+-----------------+
|    Supabase Hosted PostgreSQL     |
| (Concepts, Edges, Test Sessions)  |
+-----------------------------------+
```
"""

from __future__ import annotations
import logging
from typing import AsyncGenerator
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


# Asynchronous PostgreSQL Engine Instance configured for Supabase
engine: AsyncEngine = create_async_engine(
    cognitive_settings.database_url,
    echo=(cognitive_settings.app_env == "development"),
    pool_pre_ping=True,
    pool_size=cognitive_settings.db_pool_size,
    max_overflow=cognitive_settings.db_max_overflow,
    pool_timeout=cognitive_settings.db_pool_timeout,
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
    Bootstraps all declared relational tables in the target Supabase database.
    Called automatically during application lifespan startup event.
    """
    logger.info("Initializing relational database schema against Supabase...")
    async with engine.begin() as conn:
        # Import models inside function to ensure complete metadata registration
        import app.models.cognitive_domain_models  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Relational schema verified and initialized.")
