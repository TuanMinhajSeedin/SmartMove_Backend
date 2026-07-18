#!/usr/bin/env python3
"""Sample script to connect to Neo4j Aura.

Set these environment variables before running:
  NEO4J_URI       e.g. neo4j+s://<id>.databases.neo4j.io
  NEO4J_USER      defaults to neo4j
  NEO4J_PASSWORD  Aura password
  NEO4J_DATABASE  defaults to neo4j

Install dependency:
  pip install neo4j

Run:
  python neo4j_ingest/connect_neo4j_aura.py
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase

from aura_client import get_aura_config

load_dotenv()


def main() -> int:
    config = get_aura_config()
    uri = config.uri
    user = config.user
    password = config.password
    database = config.database

    if not uri or not password:
        print(
            "NEO4J_URI and NEO4J_PASSWORD are required environment variables.",
            file=sys.stderr,
        )
        return 1

    print(f"Connecting to Neo4j Aura at {uri} (db={database}, user={user})...")

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
            with driver.session(database=database) as session:
                record = session.run("RETURN 1 AS ok").single()
                ok = record["ok"] if record is not None else None
                if ok == 1:
                    print("Neo4j Aura connectivity verified successfully.")
                    return 0
                print("Connected to Aura, but query returned an unexpected value:", ok)
                return 1
    except Exception as exc:  # noqa: BLE001
        print("Failed to connect to Neo4j Aura:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
