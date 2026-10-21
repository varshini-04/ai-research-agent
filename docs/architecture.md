# Architecture

## Request flow

1. `Agent.invoke(query, session_id)` loads the previous run for the session (conversation history) and starts the state machine.
2. **prepare**: adds the system prompt (first turn only) and the new user message.
3. **llm**: calls the chat model with the registered tools. If it returns tool calls, go to **tools**; otherwise finish.
4. **tools**: executes each call. Unknown tools, bad arguments and tool exceptions come back as `Error: ...` tool messages so the model can recover. Then back to **llm**.
5. After `max_tool_rounds` (default 6) tools are withheld and the model must answer.
6. `Agent.ask()` then makes one extra call with `response_format=AgentAnswer` to produce validated JSON.

Each step stores a deep-copied snapshot in the `Run`, so a run can be replayed or inspected (`run.snapshots`).

## Tools

| Tool | Purpose | Backing service |
| --- | --- | --- |
| `retrieve_game` | Top-3 semantic matches with name, platform, year, genre, publisher, description | ChromaDB + OpenAI embeddings |
| `evaluate_retrieval` | LLM-as-judge returning `{useful, description}` (structured output) | OpenAI chat |
| `game_web_search` | Titles, URLs and snippets for the top results | Tavily |

Expected policy (enforced by the system prompt in `prompts.py`): retrieve, evaluate, and only search the web when `useful` is false.

## Data

Each `data/games/*.json` has `Name, Platform, Genre, Publisher, Description, YearOfRelease`. The embedded text is `[Platform] Name (Year) - Genre - Description`; the full record is stored as metadata. Ingestion validates required fields.
