"""
===============================================================================
COGNITIVE ENGINE CONFIGURATION & PARAMETER REGISTRY
===============================================================================

This module manages all configuration parameters, environment bindings, and 
hyperparameters governing cognitive algorithms, memory stability modeling, 
vector indexing, and multimodal LLM inference gateways.

Mathematical Foundations & Default Hyperparameters:
---------------------------------------------------
1. Free Spaced Repetition Scheduler (FSRS v4.5):
   - Request Retention Target (r_target): 0.75 (Desirable Difficulty Zone)
   - Initial Stability Default (S_0): 1.0 day
   - Initial Difficulty Default (D_0): 5.0 (on scale 1.0 to 10.0)

2. Yerkes-Dodson Effort Function Weights:
   - Target Effort Latency (tau_star): 120.0 seconds
   - Effort Dispersion (sigma_tau): 45.0 seconds
   - Burnout Decay Factor (kappa): 0.05

3. Maximal Marginal Relevance (MMR) Document Alignment:
   - Diversity Tradeoff (lambda_mmr): 0.70

Academic Citations:
-------------------
- Ye, J., et al. (2024). "Optimizing Spaced Repetition Schedules via Recurrent 
  Neural Memory Decay Models." Journal of Artificial Intelligence in Education.
- Bjork, R. A. (1994). "Memory and metamemory considerations in the training of 
  human beings." In J. Metcalfe & A. Shimamura (Eds.), Metacognition: Knowing 
  about knowing (pp. 185-205). MIT Press.
- Carbonell, J., & Goldstein, J. (1998). "The use of MMR, diversity-based 
  reranking for reordering documents and producing summaries." In Proceedings 
  of ACM SIGIR (pp. 335-336).
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
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_request_timeout: float = 120.0

    # Azure OpenAI Service Configuration (Enterprise cloud fallback)
    azure_openai_endpoint: str = ""
    azure_openai_key: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_embed_deployment: str = "text-embedding-3-small"
    azure_openai_api_version: str = "2024-02-01"

    # -------------------------------------------------------------------------
    # Relational Persistence (Supabase / PostgreSQL)
    # -------------------------------------------------------------------------
    supabase_url: str = ""
    supabase_anon_key: str = ""
    # Async connection string for SQLAlchemy (asyncpg driver)
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/superlearn"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: float = 30.0

    # -------------------------------------------------------------------------
    # Vector Indexing Engine (Embedded Pure Python Qdrant)
    # -------------------------------------------------------------------------
    # Running embedded pure Python mode avoids Docker dependencies
    qdrant_storage_path: str = "./qdrant_embedded_storage"
    qdrant_collection_name: str = "cognitive_knowledge_corpus"
    vector_dimension: int = 768  # Standard dimension for nomic-embed-text

    # -------------------------------------------------------------------------
    # Cognitive Science Hyperparameters & Algorithmic Weights
    # -------------------------------------------------------------------------
    # FSRS Desirable Difficulty Recall Target: R(t) in [0.70, 0.80]
    fsrs_target_retention: float = 0.75
    fsrs_stability_growth_factor: float = 0.50
    fsrs_difficulty_decay_factor: float = 0.10

    # Maximal Marginal Relevance (MMR) Parameter
    mmr_lambda_tradeoff: float = 0.70  # Balance: 0.7 relevance, 0.3 diversity

    # Yerkes-Dodson Optimal Effort Parameters
    yerkes_dodson_optimal_seconds: float = 120.0
    yerkes_dodson_dispersion: float = 45.0
    yerkes_dodson_burnout_penalty: float = 0.05

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
