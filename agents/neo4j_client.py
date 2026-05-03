"""Neo4j Aura connection helpers used by the SmartMove agents."""

from __future__ import annotations

import json
import os

from neo4j import GraphDatabase


def _neo4j_config() -> tuple[str, str, str, str]:
    """Read Aura / local Neo4j settings from environment (`NEO4J_*`)."""

    def _v(key: str, default: str = "") -> str:
        raw = os.getenv(key, default) or ""
        return raw.strip().strip('"').strip("'")

    uri = _v("NEO4J_URI")
    user = _v("NEO4J_USER", "neo4j") or "neo4j"
    password = _v("NEO4J_PASSWORD")
    database = _v("NEO4J_DATABASE", "neo4j") or "neo4j"
    return uri, user, password, database


def execute_neo4j_query(cypher: str) -> str:
    """Run Cypher against Neo4j (Aura) and return a JSON-encoded result string."""
    q = (cypher or "").strip()
    if not q:
        return "No Cypher query to run."

    uri, user, password, database = _neo4j_config()
    if not uri or not password:
        return (
            "Neo4j not configured: set NEO4J_URI and NEO4J_PASSWORD in your environment "
            "(e.g. .env at the SmartMove repo root)."
        )

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            result = session.run(q)
            rows = [r.data() for r in result]
    except Exception as e:
        return f"Neo4j query error: {e}"
    finally:
        driver.close()

    if not rows:
        return "[]"
    return json.dumps(rows, ensure_ascii=False, default=str)


def execute_neo4j_safe(cypher: str) -> str:
    """Backwards-compatible alias for `execute_neo4j_query`."""
    return execute_neo4j_query(cypher)
