"""Convenience launcher: `python run.py` starts the FastAPI server with reload."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SmartMove FastAPI server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable hot-reload (useful in production).",
    )
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parent
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        reload_dirs=[str(backend_dir)] if not args.no_reload else None,
    )


if __name__ == "__main__":
    main()
