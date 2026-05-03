# SmartMove Backend (FastAPI + LangGraph agents)

Self-contained FastAPI service that powers the Next.js frontend.
**No dependency on the repo's `trails/` directory** — every agent module lives
inside this folder.

## Layout

```
Frondend/backend/
├── main.py              # FastAPI app + endpoints
├── run.py               # `python run.py` entry point (dev: hot reload)
├── agent_cli.py         # `python agent_cli.py` for terminal-only chat
├── requirements.txt
├── README.md
└── agents/              # SmartMove LangGraph package
    ├── __init__.py      # auto-loads .env, re-exports public API
    ├── llm.py           # ChatOpenAI factory (LLM_MODEL / LLM_TEMPERATURE env)
    ├── i18n.py          # SUPPORTED_LANGS, _t, detect_language, to_english
    ├── state.py         # SmartMoveState TypedDict + STATE_FIELDS tuple
    ├── extraction.py    # regex + LLM extraction, normalization, merge
    ├── cypher.py        # Cypher generation for (:Place)-[:Schedule]->(:Place)
    ├── neo4j_client.py  # Aura config + execute_neo4j_query
    ├── nodes.py         # LangGraph node functions (intent, follow-up, ...)
    ├── graph.py         # build_app() — wires every node together
    ├── sinhala.py       # SINHALA_KEYBOARD_ROWS + Singlish→Sinhala via OpenAI
    └── cli.py           # run_cli() — interactive terminal loop
```

## Environment

`agents/__init__.py` walks up from the package directory until it finds a
`.env` file, so the agent works the same whether you launch it from the
backend folder, the repo root, or anywhere else.

Search order:
1. `Frondend/backend/.env`
2. `Frondend/.env`
3. `<repo-root>/.env`  ← typical location
4. `cwd/.env`

Required keys:

```
OPENAI_API_KEY=sk-...
NEO4J_URI=neo4j+s://<id>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
# Optional
NEO4J_DATABASE=neo4j
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.2
```

## Install

```powershell
.\.venv\Scripts\Activate.ps1
uv pip install -r Frondend\backend\requirements.txt
```

## Run

Pick **one** of these:

```powershell
# Recommended — explicit launcher
python Frondend\backend\run.py

# Direct main.py (also handles reload internally)
python Frondend\backend\main.py

# Or the uvicorn CLI
cd Frondend\backend
uvicorn main:app --reload
```

All three commands listen on `http://127.0.0.1:8000`. The Next.js dev server
proxies `/api/*` to that port (see `Frondend/next.config.mjs`).

## Endpoints

| Method | Path                      | Purpose                                              |
|-------:|---------------------------|------------------------------------------------------|
| `POST` | `/api/chat`               | Send a user message (creates a thread if missing).   |
| `POST` | `/api/resume`             | Resume a paused thread with follow-up answers.       |
| `POST` | `/api/singlish`           | Singlish → Sinhala conversion via OpenAI.            |
| `GET`  | `/api/keyboard`           | Sinhala on-screen keyboard rows.                     |
| `GET`  | `/api/state/{thread_id}`  | Last cached state (without messages) for a thread.   |
| `GET`  | `/api/health`             | Health summary (openai key + neo4j configured).      |
| `GET`  | `/api/neo4j/ping`         | Live Aura connectivity test.                         |

## Standalone CLI

For quick iteration without the web stack:

```powershell
python Frondend\backend\agent_cli.py
```

This starts the same LangGraph pipeline in a REPL, prompts for missing fields
when an interrupt fires, and prints the full agent state after each turn.

## Importing the agent package

```python
from agents import build_app, SmartMoveState, _neo4j_config, singlish_to_sinhala

app = build_app()
out = app.invoke({"messages": [...], "user_query": "Bus to Kandy from Colombo"}, config=...)
```

Public names re-exported by `agents/__init__.py`:

- `build_app`, `run_cli`, `prompt_for_missing_fields`
- `SmartMoveState`, `STATE_FIELDS`
- `SUPPORTED_LANGS`, `_t`, `detect_language`, `to_english`
- `get_llm`, `OPENAI_MODEL`, `OPENAI_TEMPERATURE`
- `SINHALA_KEYBOARD_ROWS`, `singlish_to_sinhala`, `singlish_cache_key`
- `cypher_generation_agent`, `cypher_generator_node`, `generate_cypher_for_transport`
- `execute_neo4j_query`, `execute_neo4j_safe`, `_neo4j_config`
- `validate_mandatory_fields`
