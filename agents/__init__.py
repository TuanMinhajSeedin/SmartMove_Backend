"""SmartMove agents (LangGraph human-in-the-loop pipeline).

Self-contained inside `Frondend/backend/agents/`. Auto-loads the repo-root
`.env` file at import time so `OPENAI_API_KEY` and `NEO4J_*` are available
even when the package is consumed by tests, scripts, or the FastAPI server.

Public API:
    build_app, run_cli, prompt_for_missing_fields
    SmartMoveState, STATE_FIELDS
    SUPPORTED_LANGS, _t, detect_language, to_english
    get_llm
    SINHALA_KEYBOARD_ROWS, singlish_to_sinhala, singlish_cache_key
    cypher_generation_agent, cypher_generator_node, generate_cypher_for_transport
    execute_neo4j_query, execute_neo4j_safe, _neo4j_config
    validate_mandatory_fields
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def _autoload_env() -> Path | None:
    """Walk up from this package until we find a `.env`; load whichever we hit first.

    Search order:
      1. <package>/../.env             (Frondend/backend/.env, optional)
      2. <package>/../../.env          (Frondend/.env, optional)
      3. <package>/../../../.env       (repo root .env — typical location)
      4. CWD .env                      (last-resort cwd fallback)
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / ".env",
        here.parent.parent / ".env",
        here.parent.parent.parent / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return env_path
    load_dotenv(override=False)
    return None


_LOADED_ENV = _autoload_env()


from .cypher import (  # noqa: E402  (import after env load)
    cypher_generation_agent,
    cypher_generator_node,
    generate_cypher_for_transport,
)
from .graph import build_app  # noqa: E402
from .i18n import SUPPORTED_LANGS, _t, detect_language, to_english  # noqa: E402
from .llm import OPENAI_MODEL, OPENAI_TEMPERATURE, get_llm  # noqa: E402
from .neo4j_client import (  # noqa: E402
    _neo4j_config,
    execute_neo4j_query,
    execute_neo4j_safe,
)
from .nodes import validate_mandatory_fields  # noqa: E402
from .sinhala import (  # noqa: E402
    SINHALA_KEYBOARD_ROWS,
    singlish_cache_key,
    singlish_to_sinhala,
)
from .state import STATE_FIELDS, SmartMoveState  # noqa: E402

try:  # CLI is optional — only useful for terminal usage.
    from .cli import prompt_for_missing_fields, run_cli  # noqa: E402
except Exception:  # pragma: no cover - extremely defensive
    prompt_for_missing_fields = None  # type: ignore[assignment]
    run_cli = None  # type: ignore[assignment]


__all__ = [
    "build_app",
    "run_cli",
    "prompt_for_missing_fields",
    "SmartMoveState",
    "STATE_FIELDS",
    "SUPPORTED_LANGS",
    "_t",
    "detect_language",
    "to_english",
    "get_llm",
    "OPENAI_MODEL",
    "OPENAI_TEMPERATURE",
    "SINHALA_KEYBOARD_ROWS",
    "singlish_to_sinhala",
    "singlish_cache_key",
    "cypher_generation_agent",
    "cypher_generator_node",
    "generate_cypher_for_transport",
    "execute_neo4j_query",
    "execute_neo4j_safe",
    "_neo4j_config",
    "validate_mandatory_fields",
]
