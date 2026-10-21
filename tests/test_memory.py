import pytest

from research_agent.memory import SessionNotFoundError, ShortTermMemory


def test_sessions_are_isolated_and_copied():
    mem = ShortTermMemory()
    mem.create_session("a")
    payload = {"x": 1}
    mem.add(payload, "a")
    payload["x"] = 2
    assert mem.get_last_object("a") == {"x": 1}
    assert mem.get_last_object("default") is None


def test_unknown_session_and_default_protection():
    mem = ShortTermMemory()
    with pytest.raises(SessionNotFoundError):
        mem.get_all_objects("nope")
    with pytest.raises(ValueError):
        mem.delete_session("default")
