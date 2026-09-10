import logging
from pathlib import Path

from loguru import logger
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
# The BM25 artifact is copied alongside this module in both local and Docker
# deployments. Building its path from the module location avoids assuming the
# source checkout is mounted at /backend inside the container.
BM25_INDEX_PATH = Path(__file__).resolve().parent / "bm25_rules.pkl"

LOGS_DIR = BASE_DIR / "logs"
if not LOGS_DIR.exists():
    LOGS_DIR.mkdir()

logging.basicConfig(
    level=logging.INFO,  # For displaying the default model calling logs
)

logger.add(
    sink=LOGS_DIR / "api_{time:YYYYMMDD}.log",
    level="INFO",
    rotation="00:00",
    retention="7 days",
    compression="zip",
)


class Settings(BaseSettings):
    api_key: SecretStr = Field(
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        alias="OPENROUTER_API_KEY",
    )
    pinecone_api_key: SecretStr = Field(alias="PINECONE_API_KEY")
    tavily_api_key: str
    model_provider: str = "openai"
    model_names: list[str] = ["inclusionai/ling-3.0-flash-fin:free"]
    model_base_url: str | None = "https://openrouter.ai/api/v1"
    embeddings_model_name: str = "openai/text-embedding-3-small"
    embeddings_base_url: str | None = "https://openrouter.ai/api/v1"
    bm25_index_path: Path = Field(default=BM25_INDEX_PATH)
    pinecone_index_name: str
    pinecone_namespace: str | None = None
    token_bearer_url: str = "/api/v1/auth/login"
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_expiry_mins: int = 1500
    refresh_token_expiry_days: int = 1
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_database: str
    pgvector_collection_name: str

    @property
    def database_uri(self) -> str:
        """Generate PostgreSQL connection string for sqlalchemy."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    @property
    def checkpointer_uri(self) -> str:
        """Generate PostgreSQL connection string for checkpointer."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}?sslmode=disable"

    @property
    def pgvector_connection(self) -> str:
        """Generate PostgreSQL connection string for PGVector."""
        return f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="allow")


settings = Settings()  # type: ignore
