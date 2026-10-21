"""Environment-driven configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_base_url: str | None
    chat_model: str
    embedding_model: str
    tavily_api_key: str | None
    chroma_path: Path
    collection: str
    data_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
            chroma_path=Path(os.getenv("CHROMA_PATH", ".chroma")),
            collection=os.getenv("CHROMA_COLLECTION", "games"),
            data_dir=Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data" / "games"))),
        )

    def openai_client(self) -> OpenAI:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env.")
        return OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
