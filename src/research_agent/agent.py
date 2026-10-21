"""Tool-calling agent built on the state machine."""

from __future__ import annotations

import json
from typing import Literal, TypedDict

from openai import OpenAI
from pydantic import BaseModel, Field

from research_agent.llm import LLM
from research_agent.memory import ShortTermMemory
from research_agent.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from research_agent.state_machine import EntryPoint, Run, StateMachine, Step, Termination
from research_agent.tooling import Tool, ToolCall

DEFAULT_MAX_TOOL_ROUNDS = 6


class AgentState(TypedDict, total=False):
    user_query: str
    instructions: str
    messages: list[BaseMessage]
    current_tool_calls: list[ToolCall] | None
    total_tokens: int
    tool_rounds: int
    session_id: str


class AgentAnswer(BaseModel):
    """Structured final answer."""

    answer: str = Field(description="Direct answer to the user's question.")
    source: Literal["knowledge_base", "web", "none"] = Field(
        description="Where the evidence came from."
    )
    citations: list[str] = Field(
        default_factory=list, description="Game names from the knowledge base and/or URLs."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence given the evidence found."
    )


class Agent:
    def __init__(
        self,
        instructions: str,
        tools: list[Tool] | None = None,
        model_name: str | None = None,
        temperature: float = 0.0,
        client: OpenAI | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    ):
        self.instructions = instructions
        self.tools = tools or []
        self.max_tool_rounds = max_tool_rounds
        self.memory = ShortTermMemory()
        self.llm = LLM(model=model_name, temperature=temperature, tools=self.tools, client=client)
        self.workflow = self._build_workflow()

    # --- steps -----------------------------------------------------------------------------

    def _prepare(self, state: AgentState) -> AgentState:
        messages = list(state.get("messages") or [])
        if not messages:
            messages = [SystemMessage(content=state["instructions"])]
        messages.append(UserMessage(content=state["user_query"]))
        return {"messages": messages, "tool_rounds": 0}

    def _call_llm(self, state: AgentState) -> AgentState:
        # Out of tool budget: ask for a final answer without offering tools.
        exhausted = state.get("tool_rounds", 0) >= self.max_tool_rounds
        llm = self.llm if not exhausted else LLM(
            model=self.llm.model, temperature=self.llm.temperature, client=self.llm.client
        )
        response = llm.invoke(state["messages"])
        tool_calls = None if exhausted else response.tool_calls
        total = state.get("total_tokens", 0)
        if response.token_usage:
            total += response.token_usage.total_tokens
        ai_message = AIMessage(content=response.content, tool_calls=tool_calls)
        return {
            "messages": state["messages"] + [ai_message],
            "current_tool_calls": tool_calls,
            "total_tokens": total,
        }

    def _run_tools(self, state: AgentState) -> AgentState:
        registry = {t.name: t for t in self.tools}
        results = []
        for call in state.get("current_tool_calls") or []:
            name = call.function.name
            results.append(
                ToolMessage(
                    tool_call_id=call.id, name=name, content=self._execute(registry, call)
                )
            )
        return {
            "messages": state["messages"] + results,
            "current_tool_calls": None,
            "tool_rounds": state.get("tool_rounds", 0) + 1,
        }

    @staticmethod
    def _execute(registry: dict[str, Tool], call: ToolCall) -> str:
        """Run one tool call. Errors are returned to the model rather than raised."""
        tool = registry.get(call.function.name)
        if tool is None:
            return f"Error: unknown tool '{call.function.name}'."
        try:
            args = json.loads(call.function.arguments or "{}")
            return str(tool(**args))
        except Exception as exc:  # noqa: BLE001 - surfaced to the model
            return f"Error running {call.function.name}: {exc}"

    def _build_workflow(self) -> StateMachine[AgentState]:
        machine = StateMachine[AgentState](AgentState)
        entry = EntryPoint[AgentState]()
        prepare = Step[AgentState]("prepare", self._prepare)
        llm_step = Step[AgentState]("llm", self._call_llm)
        tools_step = Step[AgentState]("tools", self._run_tools)
        end = Termination[AgentState]()

        machine.add_steps([entry, prepare, llm_step, tools_step, end])
        machine.connect(entry, prepare)
        machine.connect(prepare, llm_step)
        machine.connect(
            llm_step,
            [tools_step, end],
            lambda s: tools_step if s.get("current_tool_calls") else end,
        )
        machine.connect(tools_step, llm_step)
        return machine

    # --- public API ------------------------------------------------------------------------

    def invoke(self, query: str, session_id: str | None = None) -> Run[AgentState]:
        """Run the agent on ``query``; conversation context is kept per ``session_id``."""
        session_id = session_id or "default"
        self.memory.create_session(session_id)

        previous: list[BaseMessage] = []
        last = self.memory.get_last_object(session_id)
        if last and (final := last.get_final_state()):
            previous = final["messages"]

        run = self.workflow.run(
            {
                "user_query": query,
                "instructions": self.instructions,
                "messages": previous,
                "current_tool_calls": None,
                "session_id": session_id,
            }
        )
        self.memory.add(run, session_id)
        return run

    def ask(self, query: str, session_id: str | None = None) -> AgentAnswer:
        """Run the agent, then distil the conversation into a validated ``AgentAnswer``."""
        run = self.invoke(query, session_id)
        messages = run.get_final_state()["messages"]
        formatter = LLM(model=self.llm.model, temperature=0.0, client=self.llm.client)
        prompt = UserMessage(
            content="Using only the conversation above, produce the final structured answer "
            "to the user's latest question."
        )
        reply = formatter.invoke([*messages, prompt], response_format=AgentAnswer)
        return AgentAnswer.model_validate_json(reply.content)

    def final_text(self, run: Run[AgentState]) -> str:
        return run.get_final_state()["messages"][-1].content or ""

    def reset_session(self, session_id: str | None = None) -> None:
        self.memory.reset(session_id)
