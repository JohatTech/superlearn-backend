"""
===============================================================================
UNIT TESTS FOR LOCAL SQLITE FALLBACK & PERSISTENCE MANAGER
===============================================================================
"""

import os
import pytest
from app.core.cognitive_config import cognitive_settings
from app.db.supabase_persistence_manager import switch_to_sqlite_fallback, is_postgres


@pytest.mark.asyncio
async def test_use_local_sqlite_env_override(monkeypatch):
    """Test that setting USE_LOCAL_SQLITE=true forces sqlite+aiosqlite database URL."""
    monkeypatch.setenv("USE_LOCAL_SQLITE", "true")
    url = cognitive_settings.async_database_url
    assert "sqlite+aiosqlite" in url
    assert "superlearn.db" in url


@pytest.mark.asyncio
async def test_switch_to_sqlite_fallback_rebinds_engine():
    """Test that switch_to_sqlite_fallback dynamically rebinds the persistence manager."""
    switch_to_sqlite_fallback(reason="Unit test verification")
    from app.db import supabase_persistence_manager
    assert supabase_persistence_manager.is_postgres is False
    assert "sqlite" in supabase_persistence_manager.db_url
