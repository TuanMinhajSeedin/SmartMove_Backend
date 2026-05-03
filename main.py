"""
FastAPI backend that exposes the SmartMove LangGraph (HITL) app to the Next.js frontend.

This module is **self-contained** — all agent code lives next to it under
`agents/` and there is no dependency on `trails/`.

Endpoints
---------
POST /api/chat         -> send a new user message (creates thread if none).
POST /api/resume       -> resume a thread that's waiting on a follow-up interrupt.
POST /api/singlish     -> convert Singlish text to Sinhala via OpenAI.
GET  /api/keyboard     -> Sinhala on-screen keyboard rows.
GET  /api/state/{id}   -> last cached state (without messages) for a thread.
GET  /api/health       -> healthcheck (includes openai + neo4j status).
GET  /api/neo4j/ping   -> live Neo4j Aura connectivity test.

Environment
-----------
The `agents` package autoloads `.env` on import (typically next to `main.py`, or one/two dirs up). Required keys:
  - OPENAI_API_KEY
  - NEO4J_URI         (e.g. neo4j+s://<id>.databases.neo4j.io)
  - NEO4J_USER        (defaults to "neo4j" — fine for Aura)
  - NEO4J_PASSWORD
Optional:
  - NEO4J_DATABASE    (defaults to "neo4j")
  - LLM_MODEL         (defaults to "gpt-4o-mini")
  - LLM_TEMPERATURE   (defaults to 0.2)
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from neo4j import GraphDatabase

from agents import (
    SINHALA_KEYBOARD_ROWS,
    _neo4j_config,
    _t,
    build_app,
    detect_language,
    singlish_to_sinhala,
)

BACKEND_DIR = Path(__file__).resolve().parent


def _resolve_dotenv_path() -> Path:
    """Where `.env` lives for this deployment (same walk as `agents._autoload_env`, from `main.py` dir)."""
    here = BACKEND_DIR
    for env_path in (
        here / ".env",
        here.parent / ".env",
        here.parent.parent / ".env",
    ):
        if env_path.exists():
            return env_path
    return here / ".env"


ENV_PATH = _resolve_dotenv_path()


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("smartmove.api")


def _has_openai_key() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def _neo4j_ping() -> dict[str, Any]:
    """Open a short-lived driver and run RETURN 1 to verify Aura connectivity."""
    uri, user, password, database = _neo4j_config()
    if not uri or not password:
        return {
            "ok": False,
            "configured": False,
            "uri": uri or None,
            "database": database,
            "error": "NEO4J_URI / NEO4J_PASSWORD not set in .env",
        }
    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            rec = session.run("RETURN 1 AS ok").single()
            value = rec["ok"] if rec else None
        return {
            "ok": value == 1,
            "configured": True,
            "uri": uri,
            "user": user,
            "database": database,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "configured": True,
            "uri": uri,
            "user": user,
            "database": database,
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:  # noqa: BLE001
                pass


# ---- Startup checks ---------------------------------------------------------
log.info("Loading env from %s (exists=%s)", ENV_PATH, ENV_PATH.exists())
if _has_openai_key():
    log.info("OPENAI_API_KEY: present")
else:
    log.warning("OPENAI_API_KEY missing — language / extraction / response LLM calls will fall back.")

_ping = _neo4j_ping()
if _ping.get("ok"):
    log.info("Neo4j Aura connected: uri=%s db=%s user=%s", _ping["uri"], _ping["database"], _ping["user"])
else:
    log.warning("Neo4j connectivity check failed: %s", _ping.get("error") or _ping)
# ----------------------------------------------------------------------------


app = FastAPI(title="SmartMove API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GRAPH_APP = build_app()
LAST_STATE: dict[str, dict[str, Any]] = {}


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message text")
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    updates: dict[str, str]


class SinglishRequest(BaseModel):
    text: str


def _serialise_state(out: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in out.items() if k != "messages" and k != "__interrupt__"}


def _build_initial_state(text: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=text)],
        "user_query": text,
        "user_query_original": text,
        "language": None,
        "intent": None,
        "origin": None,
        "destination": None,
        "departure_time": None,
        "date": None,
        "transport_type": None,
        "fare": None,
        "extracted_data": None,
        "missing_fields": None,
        "cypher_query": None,
        "result": None,
        "response": None,
        "follow_up_question": None,
    }


_LABEL_KEYS = {
    "origin": "Origin",
    "destination": "Destination",
    "departure_time": "Departure time",
    "fare": "Fare",
}


def _format_response(thread_id: str, out: dict[str, Any]) -> dict[str, Any]:
    state_view = _serialise_state(out)
    LAST_STATE[thread_id] = state_view
    lang = out.get("language") or "en"

    if "__interrupt__" in out:
        payload = out["__interrupt__"][0].value
        missing = list(payload.get("missing_fields") or [])
        labels = {f: _t(lang, _LABEL_KEYS.get(f, f)) for f in missing}

        # `fare` is a toggle in the UI: off -> the agent should receive "no",
        # on  -> a value (cheapest, max LKR X, or generic "yes") is required.
        toggleable = ["fare"] if "fare" in missing else []
        toggle_meta = {
            "fare": {
                "prompt": _t(lang, "fare_toggle"),
                "need_value": _t(lang, "fare_toggle_need_value"),
                "input_label": _t(lang, "Fare"),
                "off_value": "no",
            }
        } if "fare" in missing else {}

        return {
            "thread_id": thread_id,
            "type": "interrupt",
            "language": lang,
            "interrupt": {
                "kind": payload.get("kind", "follow_up_question"),
                "question": payload.get("question", ""),
                "missing_fields": missing,
                "labels": labels,
                "toggleable": toggleable,
                "toggles": toggle_meta,
            },
            "state": state_view,
        }

    return {
        "thread_id": thread_id,
        "type": "message",
        "language": lang,
        "response": out.get("response") or "",
        "state": state_view,
    }


@app.get("/api/health")
def healthcheck() -> dict[str, Any]:
    """Lightweight health summary; does NOT open Neo4j on every call."""
    return {
        "status": "ok",
        "openai_key": _has_openai_key(),
        "neo4j_configured": bool((os.getenv("NEO4J_URI") or "").strip())
        and bool((os.getenv("NEO4J_PASSWORD") or "").strip()),
        "env_path": str(ENV_PATH),
        "env_loaded": ENV_PATH.exists(),
    }


@app.get("/api/neo4j/ping")
def neo4j_ping() -> dict[str, Any]:
    """Live Neo4j connectivity test."""
    return _neo4j_ping()


@app.get("/api/keyboard")
def keyboard() -> dict[str, list[list[str]]]:
    return {"rows": SINHALA_KEYBOARD_ROWS}


@app.post("/api/singlish")
def to_sinhala(req: SinglishRequest) -> dict[str, str]:
    return {"sinhala": singlish_to_sinhala(req.text or "")}


@app.get("/api/state/{thread_id}")
def get_state(thread_id: str) -> dict[str, Any]:
    return {"thread_id": thread_id, "state": LAST_STATE.get(thread_id, {})}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    text = (req.message or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="message must not be empty")

    thread_id = req.thread_id or f"web-{uuid.uuid4().hex}"
    cfg = {"configurable": {"thread_id": thread_id}}

    state = _build_initial_state(text)
    state["language"] = detect_language(text)

    out = GRAPH_APP.invoke(state, cfg)
    return _format_response(thread_id, out)


@app.post("/api/resume")
def resume(req: ResumeRequest) -> dict[str, Any]:
    if not req.thread_id:
        raise HTTPException(status_code=400, detail="thread_id required")
    cfg = {"configurable": {"thread_id": req.thread_id}}
    out = GRAPH_APP.invoke(Command(resume=req.updates or {}), cfg)
    return _format_response(req.thread_id, out)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(BACKEND_DIR)],
    )
