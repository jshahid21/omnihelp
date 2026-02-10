"""
Application settings via pydantic-settings (strict requirement for enterprise config).

All secrets and env-specific values come from environment; never hardcode.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Omni-Help application settings. Loaded from env and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")

    # Tavily
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key")

    # LangSmith
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangSmith tracing")
    langchain_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    langchain_project: Optional[str] = Field(default="omni-help", description="LangSmith project name")

    # Database (SQL)
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/omnihelp.db",
        description="SQLAlchemy async URL (SQLite or PostgreSQL)",
    )

    # Vector store
    chroma_persist_directory: Optional[str] = Field(
        default="./data/chroma",
        description="ChromaDB persist directory (dev)",
    )
    qdrant_url: Optional[str] = Field(default=None, description="Qdrant URL (prod)")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")

    # Router
    router_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence below this routes to Clarification Node",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
