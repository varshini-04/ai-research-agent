from typing import TypedDict

import pytest

from research_agent.state_machine import EntryPoint, StateMachine, Step, Termination


class S(TypedDict, total=False):
    n: int


def _machine(limit: int, max_steps: int = 50) -> StateMachine[S]:
    m = StateMachine[S](S, max_steps=max_steps)
    entry, inc, end = EntryPoint[S](), Step[S]("inc", lambda s: {"n": s["n"] + 1}), Termination[S]()
    m.add_steps([entry, inc, end])
    m.connect(entry, inc)
    m.connect(inc, [inc, end], lambda s: inc if s["n"] < limit else end)
    return m


def test_loops_until_condition_and_records_snapshots():
    run = _machine(3).run({"n": 0})
    assert run.get_final_state() == {"n": 3}
    assert [s.step_id for s in run.snapshots] == ["__entry__", "inc", "inc", "inc"]
    assert run.metadata["end_timestamp"] is not None


def test_max_steps_guard():
    with pytest.raises(RuntimeError, match="max_steps"):
        _machine(10_000, max_steps=5).run({"n": 0})


def test_rejects_state_with_no_schema_fields():
    with pytest.raises(ValueError):
        _machine(1).run({"other": 1})  # type: ignore[typeddict-unknown-key]
