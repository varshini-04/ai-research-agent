from conftest import FakeOpenAI, completion
from research_agent.config import PROJECT_ROOT
from research_agent.tools.evaluate_retrieval import EvaluationReport, make_evaluate_retrieval
from research_agent.tools.retrieve_game import make_retrieve_game
from research_agent.tools.web_search import make_game_web_search
from research_agent.vectorstore import GameStore


def test_retrieve_game_formats_results(tmp_path, fake_embedding):
    store = GameStore(tmp_path / "db", "test_games", fake_embedding)
    tool = make_retrieve_game(store)
    assert tool("anything") == "No documents found."
    store.ingest(PROJECT_ROOT / "data" / "games")
    assert "Platform:" in tool("Gran Turismo")


def test_evaluate_retrieval_returns_report_json():
    reply = completion('{"useful": false, "description": "wrong platform"}')
    reply.choices[0].message.parsed = EvaluationReport(useful=False, description="wrong platform")
    tool = make_evaluate_retrieval(FakeOpenAI([reply]), "m")
    assert '"useful":false' in tool("q", "docs")


def test_web_search_without_key_degrades_gracefully():
    assert "unavailable" in make_game_web_search(None)("q")
