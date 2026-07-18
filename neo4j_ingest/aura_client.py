from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return (os.getenv(key, default) or "").strip().strip('"').strip("'")


@dataclass(frozen=True)
class Neo4jAuraConfig:
    uri: str
    user: str
    password: str
    database: str


def get_aura_config() -> Neo4jAuraConfig:
    """Read Neo4j Aura configuration from environment variables."""
    return Neo4jAuraConfig(
        uri=_env("NEO4J_URI"),
        user=_env("NEO4J_USER", "neo4j") or "neo4j",
        password=_env("NEO4J_PASSWORD"),
        database=_env("NEO4J_DATABASE", "neo4j") or "neo4j",
    )


def connect_aura(config: Optional[Neo4jAuraConfig] = None) -> Driver:
    """Connect to Neo4j Aura and verify connectivity."""
    config = config or get_aura_config()
    if not config.uri or not config.password:
        raise ValueError("NEO4J_URI and NEO4J_PASSWORD must be set in environment.")

    driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
    driver.verify_connectivity()
    return driver


def verify_aura_connection(config: Optional[Neo4jAuraConfig] = None) -> bool:
    """Return True when the Aura connection can be established."""
    try:
        with connect_aura(config) as driver:
            return True
    except Exception:
        return False
