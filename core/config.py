"""
core/config.py
Centralised configuration using Pydantic-Settings.
All values are read from environment variables / .env file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

# Always load .env before anything else so OS env vars are populated
# even when the module is imported outside of a Streamlit context.
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from .env"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = Field(default="Pennywise", description="Application name")
    app_env: str = Field(default="development", description="Environment")
    log_level: str = Field(default="INFO")

    # ── IBM Cloud ─────────────────────────────────────────────────────────────
    ibm_cloud_api_key: str = Field(
        default="", description="IBM Cloud API key"
    )

    # ── IBM WatsonX.ai ────────────────────────────────────────────────────────
    watsonx_url: str = Field(
        default="https://us-south.ml.cloud.ibm.com",
        description="WatsonX endpoint URL",
    )
    watsonx_project_id: str = Field(
        default="", description="WatsonX project ID"
    )

    # ── IBM Granite Models ────────────────────────────────────────────────────
    granite_model_id: str = Field(
        default="ibm/granite-4-h-small",
        description="Granite chat model ID",
    )
    granite_embedding_model: str = Field(
        default="ibm/granite-embedding-278m-multilingual",
        description="Granite embedding model ID",
    )

    # ── IBM Watson Language Translator ────────────────────────────────────────
    language_translator_apikey: str = Field(default="")
    language_translator_url: str = Field(default="")

    # ── IBM Watson Discovery (optional) ───────────────────────────────────────
    watson_discovery_apikey: str = Field(default="")
    watson_discovery_url: str = Field(default="")
    watson_discovery_project_id: str = Field(default="")

    # ── Language settings ─────────────────────────────────────────────────────
    default_language: str = Field(default="en")
    supported_languages: str = Field(default="en,hi,mr,ta,te,bn,gu,kn,pa")

    @property
    def supported_language_list(self) -> List[str]:
        return [lang.strip() for lang in self.supported_languages.split(",")]

    # ── RAG / Vector Store ───────────────────────────────────────────────────
    vector_store_path: str = Field(default="./data/vector_store")
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    top_k_results: int = Field(default=5)

    # ── LLM Generation ────────────────────────────────────────────────────────
    max_new_tokens: int = Field(default=1024)
    min_new_tokens: int = Field(default=50)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)
    top_k: int = Field(default=50)
    repetition_penalty: float = Field(default=1.1)

    # ── UI ────────────────────────────────────────────────────────────────────
    streamlit_theme: str = Field(default="light")

    @field_validator("ibm_cloud_api_key", mode="before")
    @classmethod
    def warn_missing_api_key(cls, v: str) -> str:
        if not v or v == "your_ibm_cloud_api_key_here":
            import warnings
            warnings.warn(
                "IBM_CLOUD_API_KEY is not set. WatsonX calls will fail. "
                "Set a real key in your .env file.",
                stacklevel=2,
            )
        return v

    @property
    def watsonx_credentials(self) -> dict:
        """Returns the credentials dict expected by ibm-watsonx-ai."""
        return {
            "url": self.watsonx_url,
            "apikey": self.ibm_cloud_api_key,
        }

    @property
    def generation_params(self) -> dict:
        """TextGenParameters dict for WatsonX."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }

    @property
    def is_discovery_enabled(self) -> bool:
        return bool(self.watson_discovery_apikey and self.watson_discovery_url)

    @property
    def is_translator_enabled(self) -> bool:
        return bool(
            self.language_translator_apikey
            and self.language_translator_url
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
