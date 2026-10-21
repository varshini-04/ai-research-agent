"""A small generic state machine used to model the agent loop as explicit steps."""

from __future__ import annotations

import copy
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar, cast, get_type_hints

logger = logging.getLogger(__name__)

StateSchema = TypeVar("StateSchema")


class Step(Generic[StateSchema]):
    def __init__(self, step_id: str, logic: Callable[[StateSchema], dict]):
        self.step_id = step_id
        self.logic = logic

    def __repr__(self) -> str:
        return f"Step('{self.step_id}')"

    def run(self, state: StateSchema, state_schema: type[StateSchema]) -> StateSchema:
        result = self.logic(state)
        expected = get_type_hints(state_schema)
        updated = {**state}
        for key, value in result.items():
            if key in expected:
                updated[key] = value
        return cast(StateSchema, updated)


class EntryPoint(Step[StateSchema]):
    def __init__(self):
        super().__init__("__entry__", lambda _: {})


class Termination(Step[StateSchema]):
    def __init__(self):
        super().__init__("__termination__", lambda _: {})


@dataclass
class Transition(Generic[StateSchema]):
    source: str
    targets: list[str]
    condition: Callable[[StateSchema], Any] | None = None

    def resolve(self, state: StateSchema) -> list[str]:
        if not self.condition:
            return self.targets
        result = self.condition(state)
        if isinstance(result, Step):
            return [result.step_id]
        if isinstance(result, list):
            return [r.step_id if isinstance(r, Step) else r for r in result]
        return [result]


@dataclass
class Snapshot(Generic[StateSchema]):
    snapshot_id: str
    timestamp: datetime
    state_data: StateSchema
    step_id: str

    @classmethod
    def create(cls, state_data: StateSchema, step_id: str) -> Snapshot[StateSchema]:
        return cls(str(uuid.uuid4()), datetime.now(), state_data, step_id)


@dataclass
class Run(Generic[StateSchema]):
    """One execution of the machine: an ordered list of state snapshots."""

    run_id: str
    start_timestamp: datetime
    snapshots: list[Snapshot[StateSchema]] = field(default_factory=list)
    end_timestamp: datetime | None = None

    @classmethod
    def create(cls) -> Run[StateSchema]:
        return cls(run_id=str(uuid.uuid4()), start_timestamp=datetime.now())

    @property
    def metadata(self) -> dict:
        fmt = "%Y-%m-%d %H:%M:%S.%f"
        return {
            "run_id": self.run_id,
            "start_timestamp": self.start_timestamp.strftime(fmt),
            "end_timestamp": self.end_timestamp.strftime(fmt) if self.end_timestamp else None,
            "snapshot_counts": len(self.snapshots),
        }

    def complete(self) -> None:
        self.end_timestamp = datetime.now()

    def get_final_state(self) -> StateSchema | None:
        return self.snapshots[-1].state_data if self.snapshots else None


class StateMachine(Generic[StateSchema]):
    def __init__(self, state_schema: type[StateSchema], max_steps: int = 50):
        self.state_schema = state_schema
        self.max_steps = max_steps
        self.steps: dict[str, Step[StateSchema]] = {}
        self.transitions: dict[str, list[Transition[StateSchema]]] = {}

    def add_steps(self, steps: list[Step[StateSchema]]) -> None:
        for step in steps:
            self.steps[step.step_id] = step

    def connect(
        self,
        source: Step[StateSchema] | str,
        targets: Step[StateSchema] | str | list[Step[StateSchema] | str],
        condition: Callable[[StateSchema], Any] | None = None,
    ) -> None:
        src = source.step_id if isinstance(source, Step) else source
        target_list = targets if isinstance(targets, list) else [targets]
        ids = [t.step_id if isinstance(t, Step) else t for t in target_list]
        self.transitions.setdefault(src, []).append(Transition(src, ids, condition))

    def run(self, state: StateSchema) -> Run[StateSchema]:
        if not set(state) & set(get_type_hints(self.state_schema)):
            raise ValueError("Initial state shares no fields with the schema.")
        entries = [s for s in self.steps.values() if isinstance(s, EntryPoint)]
        if len(entries) != 1:
            raise RuntimeError(f"Expected exactly one EntryPoint, found {len(entries)}.")

        run = Run.create()
        current = entries[0].step_id
        for _ in range(self.max_steps):
            step = self.steps[current]
            if isinstance(step, Termination):
                logger.debug("Terminating")
                run.complete()
                return run

            logger.debug("Executing step: %s", current)
            state = step.run(state, self.state_schema)
            run.snapshots.append(Snapshot.create(copy.deepcopy(state), current))

            next_steps: list[str] = []
            for transition in self.transitions.get(current, []):
                next_steps += transition.resolve(state)
            if not next_steps:
                raise RuntimeError(f"No transitions from step: {current}")
            if len(next_steps) > 1:
                raise NotImplementedError("Parallel execution is not supported.")
            current = next_steps[0]

        raise RuntimeError(f"State machine exceeded max_steps={self.max_steps}")


__all__ = ["EntryPoint", "Run", "Snapshot", "StateMachine", "Step", "Termination", "Transition"]
