"""Shared LLM factory for the SmartMove agents.

Reads model and temperature from environment variables `LLM_MODEL` and
`LLM_TEMPERATURE` (with sensible defaults). Every agent in this package goes
through `get_llm()` so the model can be swapped centrally.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def _model_name() -> str:
    return os.getenv("LLM_MODEL", "gpt-4o-mini")


def _temperature() -> float:
    try:
        return float(os.getenv("LLM_TEMPERATURE", "0.2"))
    except (TypeError, ValueError):
        return 0.2


# Backwards-compatible module-level constants; preferred path is `get_llm()` so
# changes to the env at runtime are picked up.
OPENAI_MODEL = _model_name()
OPENAI_TEMPERATURE = _temperature()


def get_llm() -> ChatOpenAI:
    """Return a fresh ChatOpenAI client honoring current env settings."""
    return ChatOpenAI(model=_model_name(), temperature=_temperature())
