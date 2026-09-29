"""
===============================================================================
COGNITIVE ENGINE CONFIGURATION & PARAMETER REGISTRY
===============================================================================

This module manages all configuration parameters, environment bindings, and 
hyperparameters governing cognitive algorithms, memory stability modeling, 
vector indexing, and multimodal LLM inference gateways.
"""

from __future__ import annotations
import os
import tempfile
from typing import Literal
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CognitiveEngineSettings(BaseSettings):
    """
    Centralized configuration registry for SuperLearn Cognitive Engine.
    Loads values from environment variables or .env file with fallback defaults.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application & Environment Runtime
    # -------------------------------------------------------------------------
    app_name: str = "SuperLearn Cognitive Engine"
    app_version: str = "1.0.0-mvp"
    app_env: Literal["development", "staging", "production"] = "development"
    app_port: int = 8000
    app_host: str = "0.0.0.0"
    cors_origins: str = "*"

    # -------------------------------------------------------------------------
    # LLM Inference Gateway Configuration
    # -------------------------------------------------------------------------
    # "ollama" for pure local inference; "azure" for enterprise cloud inference
    llm_provider: Literal["ollama", "azure"] = "ollama"

    # Ollama Local Service Configuration (Native binary execution on host)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_request_timeout: float = 120.0

    # Agentic Multi-Model Pipelines (LangChain + Ollama)
    agent_syllabus_model: str = "qwen2.5:3b-instruct"
    agent_syllabus_expansion_model: str = "qwen2.5:3b-instruct"
    agent_syllabus_module_batch_size: int = 4
    agent_syllabus_max_concurrent_expansions: int = 1
    agent_test_gen_model: str = "qwen2.5:3b-instruct"
    agent_evaluator_model: str = "qwen2.5:7b-instruct"
    agent_study_materials_model: str = "qwen2.5:3b-instruct"

    # Azure OpenAI Service Configuration (Enterprise cloud fallback)
    azure_openai_endpoint: str = ""
    azure_openai_key: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_embed_deployment: str = "text-embedding-3-small"
    azure_openai_api_version: str = "2024-02-01"

    # -------------------------------------------------------------------------
    # Mind Map OCR & Document Intelligence Engine
    # -------------------------------------------------------------------------
    # Options: "azure_di" (Azure Document Intelligence v4), "opensource" (PaddleOCR / GLM-OCR API), "auto"
    mindmap_ocr_provider: Literal["azure_di", "opensource", "auto"] = "auto"
    
    # Azure Document Intelligence (Read / Layout API v4)
    # Reads from AZURE_DI_ENDPOINT and AZURE_DI_KEY env vars
    azure_di_endpoint: str = ""
    azure_di_key: str = ""
    azure_di_api_version: str = "2024-11-30"

    @property
    def effective_azure_di_endpoint(self) -> str:
        """Resolves Azure DI endpoint from multiple possible env var names."""
        import os
        return (
            self.azure_di_endpoint
            or os.environ.get("AZURE_DI_ENDPOINT")
            or os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
            or ""
        ).strip()

    @property
    def effective_azure_di_key(self) -> str:
        """Resolves Azure DI API key from multiple possible env var names."""
        import os
        return (
            self.azure_di_key
            or os.environ.get("AZURE_DI_KEY")
            or os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
            or ""
        ).strip()

    # Open Source OCR Service URL (e.g. PaddleOCR-VL, GLM-OCR, or custom service)
    opensource_ocr_url: str = "http://127.0.0.1:8000/v1"


    # -------------------------------------------------------------------------
    # Relational Persistence (Supabase / PostgreSQL)
    # -------------------------------------------------------------------------
    supabase_url: str = ""
    supabase_anon_key: str = ""
    database_url: str = ""

    # Vercel & Supabase integration environment variable aliases
    superlearn_postgres_url: str = ""
    superlearn_postgres_url_non_pooling: str = ""
    superlearn_postgres_prisma_url: str = ""
    postgres_url: str = ""
    postgres_url_non_pooling: str = ""
    next_public_superlearn_taxgensupabase_url: str = ""
    next_public_superlearn_taxgensupabase_anon_key: str = ""
    next_public_supabase_url: str = ""
    next_public_supabase_anon_key: str = ""

    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: float = 30.0

    # -------------------------------------------------------------------------
    # Vector Indexing Engine (Embedded Pure Python Qdrant)
    # -------------------------------------------------------------------------
    qdrant_storage_path: str = "./qdrant_embedded_storage"
    qdrant_collection_name: str = "cognitive_knowledge_corpus"
    vector_dimension: int = 768  # Standard dimension for nomic-embed-text

    # -------------------------------------------------------------------------
    # Cognitive Science Hyperparameters & Algorithmic Weights
    # -------------------------------------------------------------------------
    fsrs_target_retention: float = 0.75
    fsrs_stability_growth_factor: float = 0.50
    fsrs_difficulty_decay_factor: float = 0.10

    mmr_lambda_tradeoff: float = 0.70

    yerkes_dodson_optimal_seconds: float = 120.0
    yerkes_dodson_dispersion: float = 45.0
    yerkes_dodson_burnout_penalty: float = 0.05

    @property
    def async_database_url(self) -> str:
        """
        Dynamically sanitize and format the database URL for SQLAlchemy asyncpg driver.
        Prioritizes non-pooling or direct pooler URLs provided by Supabase / Vercel integrations.
        """
        raw = (
            self.database_url
            or self.superlearn_postgres_url_non_pooling
            or self.superlearn_postgres_url
            or self.superlearn_postgres_prisma_url
            or self.postgres_url_non_pooling
            or self.postgres_url
            or os.environ.get("DATABASE_URL")
            or os.environ.get("SUPERLEARN_POSTGRES_URL_NON_POOLING")
            or os.environ.get("SUPERLEARN_POSTGRES_URL")
            or os.environ.get("SUPERLEARN_POSTGRES_PRISMA_URL")
            or os.environ.get("POSTGRES_URL")
            or ""
        ).strip()

        if not raw:
            return "sqlite+aiosqlite:///./superlearn_local.db"

        url = raw
        if "://" in url:
            scheme, rest = url.split("://", 1)
            # Remove unsupported query params for asyncpg like pgbouncer, supa, connection_limit
            if "?" in rest:
                base_part, query_part = rest.split("?", 1)
                clean_params = [
                    p for p in query_part.split("&")
                    if not any(bad in p.lower() for bad in ["pgbouncer", "supa", "connection_limit"])
                ]
                query_str = "&".join(clean_params)
                rest = f"{base_part}?{query_str}" if query_str else base_part
            url = f"postgresql+asyncpg://{rest}"
        else:
            url = f"postgresql+asyncpg://{url}"

        # Handle sslmode parameter for asyncpg compatibility
        if "sslmode=require" in url:
            url = url.replace("sslmode=require", "ssl=require")
        elif "sslmode=disable" in url:
            url = url.replace("sslmode=disable", "")

        return url.rstrip("?&")

    @property
    def effective_supabase_url(self) -> str:
        return (
            self.supabase_url
            or self.next_public_superlearn_taxgensupabase_url
            or self.next_public_supabase_url
            or os.environ.get("NEXT_PUBLIC_SUPERLEARN_TAXGENSUPABASE_URL")
            or os.environ.get("SUPABASE_URL")
            or ""
        ).strip()

    @property
    def effective_supabase_anon_key(self) -> str:
        return (
            self.supabase_anon_key
            or self.next_public_superlearn_taxgensupabase_anon_key
            or self.next_public_supabase_anon_key
            or os.environ.get("NEXT_PUBLIC_SUPERLEARN_TAXGENSUPABASE_ANON_KEY")
            or os.environ.get("SUPABASE_ANON_KEY")
            or ""
        ).strip()

    @property
    def effective_qdrant_storage_path(self) -> str:
        """
        Dynamically determine writable storage path for embedded Qdrant.
        In serverless environments (Vercel, AWS Lambda) where CWD is read-only,
        routes storage to the system's temporary directory.
        """
        if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            return os.path.join(tempfile.gettempdir(), "qdrant_embedded_storage")
        return self.qdrant_storage_path

    @model_validator(mode="after")
    def validate_production_cors(self) -> CognitiveEngineSettings:
        """Ensure wildcard CORS is never permitted in production environments."""
        if self.app_env == "production" and self.cors_origins.strip() == "*":
            raise ValueError(
                "Wildcard CORS origin ('*') is strictly forbidden in production mode. "
                "Specify explicit origin domains via CORS_ORIGINS (e.g. 'https://app.superlearn.ai')."
            )
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        """
        Parse comma-separated CORS origins into a sanitized list.
        Strips whitespace and trailing slashes for clean origin matching.
        """
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        sanitized: list[str] = []
        for origin in raw.split(","):
            cleaned = origin.strip().rstrip("/")
            if cleaned:
                sanitized.append(cleaned)
        return sanitized


# Global configuration singleton instance
cognitive_settings = CognitiveEngineSettings()
