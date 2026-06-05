from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv as _load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
RAW_RESPONSES_DIR = DATA_DIR / "raw_responses"
EXTRACTED_DIR = DATA_DIR / "extracted"
VALIDATION_DIR = DATA_DIR / "validation"
LOGS_DIR = REPO_ROOT / "logs"
CODEX_OUTPUTS_DIR = REPO_ROOT / "codex_outputs"


class Settings(BaseSettings):
    model_id: str = Field(default="gpt-5.4-nano-2026-03-17", alias="OPENAI_MODEL")
    embedding_model_id: str = "text-embedding-3-small"
    temperature: float = 0.0
    max_concurrent_requests: int = 8
    max_retries: int = 5
    max_completion_tokens: int = 800

    input_price_per_million_usd: float = 0.20
    output_price_per_million_usd: float = 1.25

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    mimic_pg_uri: str | None = Field(default=None, alias="MIMIC_PG_URI")

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)


def load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        _load_dotenv(env_path, override=False)


load_env()
SETTINGS = Settings()

MODEL_ID = SETTINGS.model_id
EMBEDDING_MODEL_ID = SETTINGS.embedding_model_id
TEMPERATURE = SETTINGS.temperature
MAX_CONCURRENT_REQUESTS = SETTINGS.max_concurrent_requests
MAX_RETRIES = SETTINGS.max_retries
MAX_COMPLETION_TOKENS = SETTINGS.max_completion_tokens
INPUT_PRICE_PER_MILLION_USD = SETTINGS.input_price_per_million_usd
OUTPUT_PRICE_PER_MILLION_USD = SETTINGS.output_price_per_million_usd
