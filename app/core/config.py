import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency guard
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False


load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)


@dataclass(slots=True)
class Settings:
    app_name: str = "Delta AI Support Troubleshooting System"
    environment: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_chat_model: str = field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"))
    openai_agent_models: str = field(default_factory=lambda: os.getenv("OPENAI_AGENT_MODELS", ""))
    openai_embedding_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    openai_embedding_dimensions: int = field(
        default_factory=lambda: int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536"))
    )

    opensearch_host: str | None = field(default_factory=lambda: os.getenv("OPENSEARCH_HOST"))
    opensearch_index: str = field(default_factory=lambda: os.getenv("OPENSEARCH_INDEX", "knowledge_base"))
    opensearch_region: str = field(default_factory=lambda: os.getenv("OPENSEARCH_REGION", "us-east-1"))
    opensearch_port: int = field(default_factory=lambda: int(os.getenv("OPENSEARCH_PORT", "443")))
    opensearch_username: str | None = field(default_factory=lambda: os.getenv("OPENSEARCH_USERNAME"))
    opensearch_password: str | None = field(default_factory=lambda: os.getenv("OPENSEARCH_PASSWORD"))
    opensearch_verify_certs: bool = field(
        default_factory=lambda: os.getenv("OPENSEARCH_VERIFY_CERTS", "true").lower() == "true"
    )
    opensearch_vector_field: str = field(default_factory=lambda: os.getenv("OPENSEARCH_VECTOR_FIELD", "embeddings"))

    ticket_api_base_url: str | None = field(default_factory=lambda: os.getenv("TICKET_API_BASE_URL"))
    ticket_api_key: str | None = field(default_factory=lambda: os.getenv("TICKET_API_KEY"))
    ticket_api_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("TICKET_API_TIMEOUT_SECONDS", "10"))
    )

    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "10")))
    aws_access_key_id: str | None = field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key: str | None = field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY"))
    dynamodb_table_name: str | None = field(default_factory=lambda: os.getenv("DYNAMODB_TABLE_NAME"))
    dynamodb_region: str = field(
        default_factory=lambda: os.getenv("DYNAMODB_REGION") or os.getenv("AWS_REGION") or os.getenv("OPENSEARCH_REGION", "us-east-1")
    )
    conversation_history_max_messages: int = field(
        default_factory=lambda: int(os.getenv("CONVERSATION_HISTORY_MAX_MESSAGES", "12"))
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
