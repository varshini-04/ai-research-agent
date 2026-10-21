"""ChromaDB-backed knowledge base of video games."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions

from research_agent.config import Settings

REQUIRED_FIELDS = ("Name", "Platform", "Genre", "Publisher", "Description", "YearOfRelease")


@dataclass(frozen=True)
class SearchResult:
    id: str
    document: str
    metadata: dict
    distance: float

    def format(self) -> str:
        m = self.metadata
        return (
            f"{m['Name']} | Platform: {m['Platform']} | Released: {m['YearOfRelease']} | "
            f"Genre: {m['Genre']} | Publisher: {m['Publisher']} | {m['Description']}"
        )


def load_games(data_dir: Path) -> list[tuple[str, dict]]:
    """Read ``*.json`` game files; the file stem is the document id."""
    games = []
    for path in sorted(Path(data_dir).glob("*.json")):
        game = json.loads(path.read_text(encoding="utf-8"))
        missing = [f for f in REQUIRED_FIELDS if f not in game]
        if missing:
            raise ValueError(f"{path.name} is missing fields: {missing}")
        games.append((path.stem, game))
    return games


def game_to_document(game: dict) -> str:
    """The text that gets embedded for a game."""
    return (
        f"[{game['Platform']}] {game['Name']} ({game['YearOfRelease']}) "
        f"- {game['Genre']} - {game['Description']}"
    )


class GameStore:
    def __init__(
        self,
        path: Path | str,
        collection: str,
        embedding_function: EmbeddingFunction,
    ):
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection, embedding_function=embedding_function
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> GameStore:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env.")
        embedder = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=settings.embedding_model,
            api_base=settings.openai_base_url,
        )
        return cls(settings.chroma_path, settings.collection, embedder)

    def count(self) -> int:
        return self._collection.count()

    def ingest(self, data_dir: Path | str) -> int:
        """Idempotently upsert every game file. Returns the number of games indexed."""
        games = load_games(Path(data_dir))
        if not games:
            raise FileNotFoundError(f"No game JSON files found in {data_dir}")
        self._collection.upsert(
            ids=[gid for gid, _ in games],
            documents=[game_to_document(g) for _, g in games],
            metadatas=[g for _, g in games],
        )
        return len(games)

    def search(self, query: str, n_results: int = 3) -> list[SearchResult]:
        if self.count() == 0:
            return []
        res = self._collection.query(query_texts=[query], n_results=n_results)
        return [
            SearchResult(id=i, document=d, metadata=m, distance=dist)
            for i, d, m, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0],
                strict=True,
            )
        ]
