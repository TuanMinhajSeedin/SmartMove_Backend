"""Optional CLI runner for the SmartMove agent (mirrors the original `run_cli`)."""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from .graph import build_app
from .i18n import _t, detect_language
from .state import SmartMoveState


def prompt_for_missing_fields(missing_fields: list[str], lang: str) -> dict[str, str]:
    label = {
        "origin": _t(lang, "Origin"),
        "destination": _t(lang, "Destination"),
        "departure_time": _t(lang, "Departure time"),
        "fare": _t(lang, "Fare"),
    }
    out: dict[str, str] = {}
    for f in missing_fields:
        while True:
            val = input(f"{label.get(f, f)}: ").strip()
            if val:
                out[f] = val
                break
    return out


def run_cli() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print(_t("en", "warn_no_key"))

    app = build_app()
    thread_id = "cli-thread"
    cfg = {"configurable": {"thread_id": thread_id}}

    print(f"{_t('en', 'smartmove')} CLI (type 'exit' to quit)\n")
    while True:
        user_text = input(f"{_t('en', 'you')}: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        state: SmartMoveState = {
            "messages": [HumanMessage(content=user_text)],
            "user_query": user_text,
            "user_query_original": user_text,
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

        out = app.invoke(state, cfg)
        while "__interrupt__" in out:
            interrupt_payload = out["__interrupt__"][0].value
            lang = out.get("language") or detect_language(user_text)
            print(f"\n{_t(lang, 'smartmove')}: {interrupt_payload.get('question')}\n")
            missing = interrupt_payload.get("missing_fields") or []
            updates = prompt_for_missing_fields(list(missing), lang)
            out = app.invoke(Command(resume=updates), cfg)

        lang = out.get("language") or detect_language(user_text)
        print(f"\n{_t(lang, 'smartmove')}: {out.get('response')}\n")
        state_view = {k: v for k, v in out.items() if k != "messages"}
        print("=== State ===")
        print(state_view)
        print("=============\n")


if __name__ == "__main__":
    run_cli()
