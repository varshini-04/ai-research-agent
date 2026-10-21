import json

import pytest

from research_agent.config import PROJECT_ROOT
from research_agent.vectorstore import GameStore, load_games

DATA = PROJECT_ROOT / "data" / "games"


def test_bundled_data_is_valid():
    games = load_games(DATA)
    assert len(games) >= 10
    assert all(isinstance(g["YearOfRelease"], int) for _, g in games)


def test_missing_field_is_rejected(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"Name": "X"}))
    with pytest.raises(ValueError, match="missing fields"):
        load_games(tmp_path)


def test_ingest_is_idempotent_and_search_finds_game(tmp_path, fake_embedding):
    store = GameStore(tmp_path / "db", "test_games", fake_embedding)
    n = store.ingest(DATA)
    store.ingest(DATA)
    assert store.count() == n
    results = store.search("Gran Turismo racing simulator", n_results=3)
    assert results[0].metadata["Name"].startswith("Gran Turismo")
    assert any("1997" in r.format() for r in results)


def test_search_empty_store(tmp_path, fake_embedding):
    assert GameStore(tmp_path / "db", "test_games", fake_embedding).search("anything") == []
