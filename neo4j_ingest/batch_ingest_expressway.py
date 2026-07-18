from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from neo4j_ingest.expressway_ingest import ExpresswayNeo4jIngester


def default_expressway_directory() -> str:
    root = Path(__file__).resolve().parent.parent
    return str(root / "ntc_time_schedule" / "Extracted" / "Done-verified" / "Expressway")


def batch_ingest_expressway(directory: str) -> int:
    print("📌 Neo4j Aura Expressway batch ingestion")
    print(f"📁 Source directory: {directory}")

    ingester = ExpresswayNeo4jIngester()
    if not ingester.connect():
        print("❌ Could not connect to Neo4j Aura. Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, and NEO4J_DATABASE.")
        return 1

    try:
        result = ingester.ingest_directory(directory)

        print("\n📊 Ingestion summary")
        print(f"  files attempted: {len(result['files'])}")
        print(f"  successful: {result['successful']}")
        print(f"  failed: {result['failed']}")
        print(f"  locations created: {result['locations_created']}")
        print(f"  trips created: {result['trips_created']}")

        if result["failed"]:
            print("\n❌ Some files failed to ingest. See details below:")
            for file_result in result["files"]:
                if not file_result["success"]:
                    print(f"  - {file_result['file']}: {file_result.get('error')}")
        return 0 if result["failed"] == 0 else 1
    finally:
        ingester.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch ingest Expressway JSON files into Neo4j Aura.")
    parser.add_argument(
        "--directory",
        default=os.getenv("EXPRESSWAY_JSON_DIR", default_expressway_directory()),
        help="Path to the Expressway JSON directory",
    )
    args = parser.parse_args()

    return batch_ingest_expressway(args.directory)


if __name__ == "__main__":
    raise SystemExit(main())
