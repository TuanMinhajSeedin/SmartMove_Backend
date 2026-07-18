from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neo4j import Driver

from neo4j_ingest.aura_client import connect_aura, get_aura_config, Neo4jAuraConfig


class ExpresswayNeo4jIngester:
    """Ingest Expressway extracted JSON files into Neo4j Aura."""

    def __init__(
        self,
        driver: Optional[Driver] = None,
        config: Optional[Neo4jAuraConfig] = None,
    ):
        self.driver = driver
        self.config = config or get_aura_config()

    def connect(self) -> bool:
        if self.driver:
            return True
        try:
            self.driver = connect_aura(self.config)
            print(f"✅ Connected to Neo4j Aura database: {self.config.database}")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"❌ Aura connection failed: {exc}", file=sys.stderr)
            return False

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def _is_location_header(self, header: str) -> bool:
        header_lower = header.lower().strip()
        location_keywords = [
            "trip",
            "number",
            "bus running",
            "running no",
            "route type",
            "bus number",
            "service no",
            "service",
        ]
        return not any(keyword in header_lower for keyword in location_keywords)

    def _normalize_location_name(self, location: str) -> str:
        normalized = location.strip()
        for suffix in [" Arrival", " Departure", " Time", " Time "]:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        return normalized.strip()

    def _extract_route_info(self, file_path: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        """Extract route_type, days, fare_type, service_type from file path.
        
        Returns: (route_type, days, fare_type, service_type)
        """
        path_parts = Path(file_path).parts
        path_str = str(file_path).lower()
        filename = Path(file_path).stem.lower()
        
        # Determine route_type: Expressway or Normal
        if "expressway" in path_str:
            route_type = "Expressway"
        elif "normalway" in path_str or "normal" in path_str:
            route_type = "Normal"
        else:
            route_type = "Unknown"
        
        # Determine fare_type: AC or Non-AC
        fare_type = None
        if "ac" in filename or "ac" in path_str:
            fare_type = "AC"
        elif "non-ac" in filename or "non-ac" in path_str or "non ac" in filename:
            fare_type = "Non-AC"
        
        # Determine service_type: Luxury, Semi Luxury, or Normal
        service_type = None
        if "luxury" in filename or "luxury" in path_str:
            if "semi" in filename or "semi" in path_str:
                service_type = "Semi Luxury"
            else:
                service_type = "Luxury"
        elif "normal" in filename or "normal" in path_str:
            service_type = "Normal"
        
        # Extract days from filename
        days = None
        if re.search(r"OddDays", filename, re.IGNORECASE):
            days = "OddDays"
        elif re.search(r"-even", filename, re.IGNORECASE):
            days = "Even"
        else:
            days_pattern = r"\(([^)]+)\)"
            days_matches = re.findall(days_pattern, filename)
            if days_matches:
                for match in days_matches:
                    cleaned = match.strip()
                    if any(day in cleaned.lower() for day in [
                        "day",
                        "friday",
                        "saturday",
                        "sunday",
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "odd",
                        "even",
                        "normal",
                    ]):
                        days = cleaned
                        break
        return route_type, days, fare_type, service_type

    def _parse_time(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, float) and str(value).lower() == "nan":
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return None
        if re.match(r"^\d{1,2}:\d{2}$", text):
            return text
        return None

    def _time_to_minutes(self, time_str: Optional[str]) -> Optional[int]:
        """Convert HH:MM time string to minutes from midnight."""
        if not time_str:
            return None
        try:
            parts = time_str.strip().split(":")
            if len(parts) != 2:
                return None
            hours = int(parts[0])
            minutes = int(parts[1])
            return hours * 60 + minutes
        except (ValueError, IndexError):
            return None

    def _create_location_nodes(self, session, locations: List[str]) -> int:
        query = """
        UNWIND $locations AS loc_name
        MERGE (place:Place {name: loc_name})
        RETURN count(place) as created
        """
        result = session.run(query, locations=locations)
        return result.single()["created"]

    def _create_trip_relationship(
        self,
        session,
        from_location: str,
        to_location: str,
        departure_time: str,
        arrival_time: str,
        route_type: str,
        days: Optional[str],
        trip_number: Optional[str],
        table_number: int,
        source_file: str,
        fare_type: Optional[str] = None,
        service_type: Optional[str] = None,
        route_type_field: Optional[str] = None,
    ) -> bool:
        # compute numeric minutes
        dep_minutes = self._time_to_minutes(departure_time)
        arr_minutes = self._time_to_minutes(arrival_time)

        # First check if an identical schedule already exists (avoid MERGE with null props)
        check_query = """
        MATCH (from:Place {name:$from_location})-[r:Schedule]->(to:Place {name:$to_location})
        WHERE r.departure_minutes = $departure_minutes
          AND r.arrival_minutes = $arrival_minutes
          AND r.route_type = $route_type
          AND COALESCE(r.days, '') = COALESCE($days, '')
          AND COALESCE(r.fare_type, '') = COALESCE($fare_type, '')
          AND COALESCE(r.service_type, '') = COALESCE($service_type, '')
          AND COALESCE(r.route_type_field, '') = COALESCE($route_type_field, '')
        RETURN r LIMIT 1
        """
        existing = session.run(
            check_query,
            from_location=from_location,
            to_location=to_location,
            departure_minutes=dep_minutes,
            arrival_minutes=arr_minutes,
            route_type=route_type,
            days=days,
            fare_type=fare_type,
            service_type=service_type,
            route_type_field=route_type_field,
        ).single()
        if existing:
            return False

        # Create relationship and set optional properties only when provided
        # Store numeric minute values in `departure_time`/`arrival_time` and keep original strings
        create_query = """
        MATCH (from:Place {name:$from_location}), (to:Place {name:$to_location})
        CREATE (from)-[schedule:Schedule {
            departure_time: $departure_minutes,
            arrival_time: $arrival_minutes,
            departure_minutes: $departure_minutes,
            arrival_minutes: $arrival_minutes,
            route_type: $route_type
        }]->(to)
        FOREACH(_ IN CASE WHEN $departure_time_str IS NULL THEN [] ELSE [1] END | SET schedule.departure_time_str = $departure_time_str)
        FOREACH(_ IN CASE WHEN $arrival_time_str IS NULL THEN [] ELSE [1] END | SET schedule.arrival_time_str = $arrival_time_str)
        FOREACH(_ IN CASE WHEN $route_type_field IS NULL THEN [] ELSE [1] END | SET schedule.route_type_field = $route_type_field)
        FOREACH(_ IN CASE WHEN $days IS NULL THEN [] ELSE [1] END | SET schedule.days = $days)
        FOREACH(_ IN CASE WHEN $fare_type IS NULL THEN [] ELSE [1] END | SET schedule.fare_type = $fare_type)
        FOREACH(_ IN CASE WHEN $service_type IS NULL THEN [] ELSE [1] END | SET schedule.service_type = $service_type)
        FOREACH(_ IN CASE WHEN $trip_number IS NULL THEN [] ELSE [1] END | SET schedule.trip_number = $trip_number)
        FOREACH(_ IN CASE WHEN $table_number IS NULL THEN [] ELSE [1] END | SET schedule.table_number = $table_number)
        FOREACH(_ IN CASE WHEN $source_file IS NULL THEN [] ELSE [1] END | SET schedule.source_file = $source_file)
        RETURN schedule
        """

        result = session.run(
            create_query,
            from_location=from_location,
            to_location=to_location,
            departure_minutes=dep_minutes,
            arrival_minutes=arr_minutes,
            departure_time_str=departure_time,
            arrival_time_str=arrival_time,
            route_type=route_type,
            route_type_field=route_type_field,
            days=days,
            fare_type=fare_type,
            service_type=service_type,
            trip_number=trip_number,
            table_number=table_number,
            source_file=source_file,
        )
        return result.single() is not None

    def _get_location_columns(self, headers: List[str]) -> Tuple[List[str], Dict[int, str]]:
        locations: List[str] = []
        index_to_location: Dict[int, str] = {}
        for idx, header in enumerate(headers):
            if self._is_location_header(header):
                normalized = self._normalize_location_name(header)
                if normalized not in locations:
                    locations.append(normalized)
                index_to_location[idx] = normalized
        return locations, index_to_location

    def _find_route_type_column(self, headers: List[str]) -> Optional[int]:
        for idx, header in enumerate(headers):
            if header and "route type" in header.lower():
                return idx
        return None

    def _parse_row_route_type(self, row: List[Any], route_type_idx: Optional[int]) -> Optional[str]:
        if route_type_idx is None or route_type_idx >= len(row):
            return None
        value = row[route_type_idx]
        if value is None:
            return None
        route_type_value = str(value).strip()
        if not route_type_value or route_type_value.lower() in {"nan", "none"}:
            return None
        return route_type_value

    def ingest_json_file(
        self,
        json_path: str,
        route_type_override: Optional[str] = None,
        days_override: Optional[str] = None,
        fare_type_override: Optional[str] = None,
        service_type_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        print(f"\n📄 Ingesting {json_path}")
        try:
            with open(json_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            return {"success": False, "error": f"Failed to load JSON: {exc}"}

        if not payload.get("success") or "extracted_data" not in payload:
            return {"success": False, "error": "Invalid JSON structure"}

        route_type, days, fare_type, service_type = self._extract_route_info(json_path)
        if route_type_override:
            route_type = route_type_override
        if days_override is not None:
            days = days_override
        if fare_type_override is not None:
            fare_type = fare_type_override
        if service_type_override is not None:
            service_type = service_type_override

        days = days or "All Days"
        tables = payload["extracted_data"].get("tables", [])
        print(f"  route_type={route_type}, days={days}, fare_type={fare_type}, service_type={service_type}, tables={len(tables)}")

        locations_created = 0
        trips_created = 0
        source_file = Path(json_path).name

        with self.driver.session(database=self.config.database) as session:
            for table in tables:
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                table_number = int(table.get("table_number", 1))

                locations, index_map = self._get_location_columns(headers)
                if len(locations) < 2:
                    print(f"  ⚠️  Skipping table {table_number}: not enough location columns")
                    continue

                locations_created += self._create_location_nodes(session, locations)

                route_type_field_idx = self._find_route_type_column(headers)
                for row in rows:
                    if not isinstance(row, list) or len(row) < 2:
                        continue

                    row_route_type = self._parse_row_route_type(row, route_type_field_idx)
                    row_values = [str(value).strip() for value in row]
                    trip_number = None
                    if route_type_field_idx is None and row_values:
                        trip_number = row_values[0] if self._parse_time(row_values[0]) is None else None

                    location_indices = [idx for idx in sorted(index_map) if idx < len(row)]
                    for from_idx, to_idx in zip(location_indices, location_indices[1:]):
                        from_location = index_map[from_idx]
                        to_location = index_map[to_idx]
                        departure_time = self._parse_time(row[from_idx])
                        arrival_time = self._parse_time(row[to_idx])
                        if not departure_time or not arrival_time:
                            continue
                        if self._create_trip_relationship(
                            session=session,
                            from_location=from_location,
                            to_location=to_location,
                            departure_time=departure_time,
                            arrival_time=arrival_time,
                            route_type=route_type,
                            days=days,
                            trip_number=trip_number,
                            table_number=table_number,
                            source_file=source_file,
                            fare_type=fare_type,
                            service_type=service_type,
                            route_type_field=row_route_type,
                        ):
                            trips_created += 1

        print(f"  ✅ Created {locations_created} locations and {trips_created} trips")
        return {
            "success": True,
            "file": json_path,
            "route_type": route_type,
            "days": days,
            "fare_type": fare_type,
            "service_type": service_type,
            "tables_processed": len(tables),
            "locations_created": locations_created,
            "trips_created": trips_created,
        }

    def ingest_directory(
        self,
        directory: str,
        pattern: str = "*_extracted_tables_english.json",
        route_type_override: Optional[str] = None,
        days_override: Optional[str] = None,
        fare_type_override: Optional[str] = None,
        service_type_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Expressway directory not found: {directory}")

        json_files = sorted(path.glob(pattern))
        if not json_files:
            return {"success": False, "error": f"No files found in {directory}"}

        summary = {
            "files": [],
            "successful": 0,
            "failed": 0,
            "locations_created": 0,
            "trips_created": 0,
        }

        for json_file in json_files:
            result = self.ingest_json_file(
                str(json_file),
                route_type_override=route_type_override,
                days_override=days_override,
                fare_type_override=fare_type_override,
                service_type_override=service_type_override,
            )
            summary["files"].append(result)
            if result["success"]:
                summary["successful"] += 1
                summary["locations_created"] += result.get("locations_created", 0)
                summary["trips_created"] += result.get("trips_created", 0)
            else:
                summary["failed"] += 1

        return summary
