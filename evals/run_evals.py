"""Live end-to-end checks against the real models.

Usage: python evals/run_evals.py   (needs OPENAI_API_KEY, TAVILY_API_KEY and a prior `ingest`)
A case passes when the answer contains one of `expect_any` (case-insensitive) and, if set,
the structured `source` matches `expect_source`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from research_agent.agent import Agent
from research_agent.config import get_settings
from research_agent.prompts import SYSTEM_INSTRUCTIONS
from research_agent.tools import build_tools


def main() -> int:
    settings = get_settings()
    cases = json.loads((Path(__file__).parent / "cases.json").read_text(encoding="utf-8"))
    agent = Agent(SYSTEM_INSTRUCTIONS, build_tools(settings), model_name=settings.chat_model)

    passed = 0
    print(f"{'result':<7}{'source':<16}question")
    for i, case in enumerate(cases):
        result = agent.ask(case["question"], session_id=f"eval-{i}")  # fresh session per case
        text = result.answer.lower()
        ok = any(s.lower() in text for s in case["expect_any"])
        if case.get("expect_source"):
            ok = ok and result.source == case["expect_source"]
        passed += ok
        print(f"{'PASS' if ok else 'FAIL':<7}{result.source:<16}{case['question']}")
    print(f"\n{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
