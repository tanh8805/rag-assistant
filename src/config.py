"""Đọc cấu hình: config.yaml (cài đặt chung) + .env (key, mật khẩu)."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")
with open(ROOT / "config.yaml", encoding="utf-8") as f:
    _yaml = yaml.safe_load(f)


@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    name: str


@dataclass
class EmbeddingConfig:
    model: str
    dim: int


@dataclass
class ChunkingConfig:
    strategy: str
    parent_size: int
    child_size: int
    child_overlap: int
    chunk_size: int
    chunk_overlap: int


@dataclass
class RetrievalConfig:
    top_k: int


@dataclass
class RerankConfig:
    model: str
    top_n: int


@dataclass
class LLMConfig:
    provider: str
    gemini_model: str
    openai_model: str


@dataclass
class Settings:
    # từ config.yaml
    database: DatabaseConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    llm: LLMConfig
    # từ .env
    postgres_password: str
    gemini_api_key: str
    openai_api_key: str

    def database_url_for(self, strategy: str) -> str:
        """Mỗi chunking strategy có database riêng: <database.name>_<strategy>."""
        db = self.database
        host = os.getenv("DB_HOST", db.host)  # trong Docker, host là "db"
        return f"postgresql+psycopg://{db.user}:{self.postgres_password}@{host}:{db.port}/{db.name}_{strategy}"

    @property
    def database_url(self) -> str:
        return self.database_url_for(self.chunking.strategy)


settings = Settings(
    database=DatabaseConfig(**_yaml["database"]),
    embedding=EmbeddingConfig(**_yaml["embedding"]),
    chunking=ChunkingConfig(**_yaml["chunking"]),
    retrieval=RetrievalConfig(**_yaml["retrieval"]),
    rerank=RerankConfig(**_yaml["rerank"]),
    llm=LLMConfig(**_yaml["llm"]),
    postgres_password=os.getenv("POSTGRES_PASSWORD", ""),
    gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
)
