"""Command line interface: ``research-agent ingest | ask | chat``."""

from __future__ import annotations

import argparse
import logging

from research_agent.agent import Agent
from research_agent.config import get_settings
from research_agent.prompts import SYSTEM_INSTRUCTIONS
from research_agent.tools import build_tools
from research_agent.vectorstore import GameStore


def _build_agent() -> Agent:
    settings = get_settings()
    return Agent(
        instructions=SYSTEM_INSTRUCTIONS,
        tools=build_tools(settings),
        model_name=settings.chat_model,
    )


def _print_answer(agent: Agent, question: str, session_id: str, as_json: bool) -> None:
    if as_json:
        print(agent.ask(question, session_id).model_dump_json(indent=2))
    else:
        print(agent.final_text(agent.invoke(question, session_id)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-agent", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="log agent steps")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="embed data/games into the local vector store")

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="return a structured JSON answer")

    chat = sub.add_parser("chat", help="interactive session with conversation memory")
    chat.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(name)s: %(message)s")
    settings = get_settings()

    if args.command == "ingest":
        count = GameStore.from_settings(settings).ingest(settings.data_dir)
        print(f"Indexed {count} games into '{settings.collection}' at {settings.chroma_path}")
        return 0

    agent = _build_agent()
    if args.command == "ask":
        _print_answer(agent, args.question, "cli", args.json)
        return 0

    print("Ask about video games (Ctrl-D or 'exit' to quit).")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if question:
            _print_answer(agent, question, "chat", args.json)
