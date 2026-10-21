import json

from conftest import FakeOpenAI, completion, tool_call
from research_agent.agent import Agent, AgentAnswer
from research_agent.tooling import tool


@tool
def echo(text: str) -> str:
    """Echo the text."""
    return f"echo:{text}"


@tool
def boom(text: str) -> str:
    """Always fails."""
    raise RuntimeError("kaput")


def test_tool_loop_and_final_answer():
    client = FakeOpenAI([
        completion(tool_calls=[tool_call("echo", {"text": "hi"})]),
        completion("done"),
    ])
    agent = Agent("be helpful", [echo], model_name="m", client=client)
    run = agent.invoke("question")

    assert agent.final_text(run) == "done"
    tool_msg = [m for m in run.get_final_state()["messages"] if m.role == "tool"][0]
    assert tool_msg.content == "echo:hi"
    # tools were offered on the first request; no local-only fields leak to the API
    assert client.calls[0]["tools"][0]["function"]["name"] == "echo"
    assert all("token_usage" not in m for m in client.calls[1]["messages"])


def test_unknown_and_failing_tools_return_errors_not_exceptions():
    client = FakeOpenAI([
        completion(tool_calls=[
            tool_call("nope", {}, "c1"),
            tool_call("boom", {"text": "x"}, "c2"),
        ]),
        completion("recovered"),
    ])
    agent = Agent("i", [boom], model_name="m", client=client)
    run = agent.invoke("q")
    contents = [m.content for m in run.get_final_state()["messages"] if m.role == "tool"]
    assert "unknown tool 'nope'" in contents[0]
    assert "kaput" in contents[1]
    assert agent.final_text(run) == "recovered"


def test_tool_round_cap_forces_final_answer():
    loop = [completion(tool_calls=[tool_call("echo", {"text": "x"}, f"c{i}")]) for i in range(2)]
    client = FakeOpenAI([*loop, completion("forced final")])
    agent = Agent("i", [echo], model_name="m", client=client, max_tool_rounds=2)
    assert agent.final_text(agent.invoke("q")) == "forced final"
    assert "tools" not in client.calls[-1]


def test_session_memory_carries_history():
    client = FakeOpenAI([completion("one"), completion("two")])
    agent = Agent("i", [], model_name="m", client=client)
    agent.invoke("first", "s")
    agent.invoke("second", "s")
    roles = [m["role"] for m in client.calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_ask_returns_validated_structured_answer():
    payload = AgentAnswer(answer="1999", source="knowledge_base", citations=["Pokemon Gold"],
                          confidence="high")
    client = FakeOpenAI([completion("1999"), completion(payload.model_dump_json())])
    answer = Agent("i", [], model_name="m", client=client).ask("when?")
    assert answer == payload
    assert json.loads(answer.model_dump_json())["source"] == "knowledge_base"
