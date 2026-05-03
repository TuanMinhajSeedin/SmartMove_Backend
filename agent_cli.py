"""Stand-alone CLI launcher for the SmartMove agent (no FastAPI / web required)."""

from __future__ import annotations

from agents import run_cli


if __name__ == "__main__":
    if run_cli is None:
        raise SystemExit("CLI runner is unavailable.")
    run_cli()
