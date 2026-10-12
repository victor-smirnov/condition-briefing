"""All configuration comes from environment variables (12-factor, Cloud Run friendly)."""

import os
from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True)
class Settings:
    model_main: str  # planning, synthesis, chat
    model_fast: str  # per-source screening and extraction
    store_url: str  # file:///path or gs://bucket/prefix
    ncbi_email: str
    ncbi_api_key: str | None
    max_workers: int


@cache
def settings() -> Settings:
    return Settings(
        model_main=os.getenv("MODEL_MAIN", "gpt-5.5"),
        model_fast=os.getenv("MODEL_FAST", "gpt-5.4-mini"),
        store_url=os.getenv("STORE_URL", "file://./data/runs"),
        ncbi_email=os.getenv("NCBI_EMAIL", "briefing-demo@example.org"),
        ncbi_api_key=os.getenv("NCBI_API_KEY") or None,
        max_workers=int(os.getenv("MAX_WORKERS", "8")),
    )
