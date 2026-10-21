from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel, Field

from research_agent.tooling import Tool, tool


class EvaluationReport(BaseModel):
    useful: bool = Field(description="Whether the documents are enough to answer the question")
    description: str = Field(description="Explanation of the evaluation result")


def make_evaluate_retrieval(client: OpenAI, model: str) -> Tool:
    @tool
    def evaluate_retrieval(question: str, retrieved_docs: str) -> str:
        """LLM-as-judge: decide whether retrieved documents can answer the question.

        Args:
            question: the original user question.
            retrieved_docs: the text returned by retrieve_game.
        Returns JSON with `useful` (bool) and `description` (why).
        """
        response = client.chat.completions.parse(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict evaluator for a retrieval-augmented system. "
                    "Say useful=true only if the documents directly contain the facts "
                    "needed (e.g. the right game AND the right platform/year).",
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nDocuments:\n{retrieved_docs}",
                },
            ],
            response_format=EvaluationReport,
        )
        return response.choices[0].message.parsed.model_dump_json()

    return evaluate_retrieval
