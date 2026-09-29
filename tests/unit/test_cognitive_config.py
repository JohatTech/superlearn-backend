"""
===============================================================================
UNIT TESTS: COGNITIVE CONFIGURATION & SECURITY VALIDATION
===============================================================================
"""

import os
import pytest
from pydantic import ValidationError
from app.core.cognitive_config import CognitiveEngineSettings


class TestCognitiveConfigSecurity:
    """Test suite verifying configuration security rules and origin sanitization."""

    def test_default_development_configuration(self):
        """Verify default settings provide secure dev defaults (not wildcard CORS)."""
        settings = CognitiveEngineSettings(
            app_env="development",
            cors_origins="http://localhost:3000",
        )
        assert settings.app_env == "development"
        assert settings.cors_origins == "http://localhost:3000"
        assert settings.cors_origins_list == ["http://localhost:3000"]

    def test_cors_origins_list_sanitization(self):
        """Verify trailing slashes and whitespace are properly stripped from CORS origins."""
        settings = CognitiveEngineSettings(
            app_env="development",
            cors_origins=" http://localhost:3000/ , https://app.superlearn.ai/ , http://127.0.0.1:8000 ",
        )
        assert settings.cors_origins_list == [
            "http://localhost:3000",
            "https://app.superlearn.ai",
            "http://127.0.0.1:8000",
        ]

    def test_cors_origins_wildcard_in_development(self):
        """Verify wildcard CORS is accepted in development mode when explicitly configured."""
        settings = CognitiveEngineSettings(
            app_env="development",
            cors_origins="*",
        )
        assert settings.cors_origins_list == ["*"]

    def test_cors_origins_wildcard_forbidden_in_production(self):
        """Verify that setting wildcard CORS in production raises a strict ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CognitiveEngineSettings(
                app_env="production",
                cors_origins="*",
            )
        assert "Wildcard CORS origin ('*') is strictly forbidden in production mode" in str(exc_info.value)

    def test_cors_origins_explicit_domain_allowed_in_production(self):
        """Verify explicit domains are permitted in production mode."""
        settings = CognitiveEngineSettings(
            app_env="production",
            cors_origins="https://app.superlearn.ai, https://api.superlearn.ai",
        )
        assert settings.app_env == "production"
        assert settings.cors_origins_list == [
            "https://app.superlearn.ai",
            "https://api.superlearn.ai",
        ]

    def test_effective_qdrant_path_serverless(self, monkeypatch):
        """Verify dynamic redirection of Qdrant storage path in serverless read-only runtimes."""
        monkeypatch.setenv("VERCEL", "1")
        settings = CognitiveEngineSettings()
        assert "qdrant_embedded_storage" in settings.effective_qdrant_storage_path
        # When VERCEL is unset, returns configured path
        monkeypatch.delenv("VERCEL")
        assert settings.effective_qdrant_storage_path == settings.qdrant_storage_path
