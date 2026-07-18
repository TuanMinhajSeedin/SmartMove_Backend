#!/usr/bin/env python3
"""
Ingest time schedule data from extracted JSON files into Neo4j
- Creates Location nodes from table headers
- Creates TRIP relationships with departure/arrival times
- Extracts route_type and days from file paths
"""

import json
import os
import re
from neo4j import GraphDatabase
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd


class TimeScheduleIngester:
    """Ingest time schedule data into Neo4j"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", 
                 password: str = "20665130@mM",
                 database: str = "timeschedules"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        
    def connect(self):
        """Connect to Neo4j database"""
        try:
            print(f"🔌 Connecting to Neo4j at {self.uri}...")
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                if result.single()["test"] == 1:
                    print(f"✅ Connected to Neo4j database: {self.database}")
                    return True
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def extract_route_info(self, file_path: str) -> Tuple[str, Optional[str]]:
        """
        Extract route type and days from file path
        Returns: (route_type, days)
        """
        path_parts = Path(file_path).parts
        route_type = None
        days = None
        
        # Extract route type from path (e.g., "Expressway", "Normal")
        # If "Normal" is in path, use "Normal" (even if Luxury is a subfolder)
        if "Expressway" in path_parts:
            route_type = "Expressway"
        elif "Normal" in path_parts:
            route_type = "Normal"  # Use "Normal" even if Luxury is a subfolder
        # Add more route types as needed
        
        # Extract days from filename
        filename = Path(file_path).stem  # Get filename without extension
        
        # First, try to extract from filename patterns like "OddDays", "Even", etc.
        # Check for OddDays or Even (case insensitive)
        if re.search(r'-OddDays', filename, re.IGNORECASE):
            days = "OddDays"
        elif re.search(r'-Even', filename, re.IGNORECASE):
            days = "Even"
        # If not found, try pattern to match days in parentheses: (Friday,Saturday & Sunday)
        # But filter out route numbers and bus types
        else:
            days_pattern = r'\(([^)]+)\)'
            days_matches = re.findall(days_pattern, filename)
            if days_matches:
                # Filter out common route identifiers and bus types
                route_identifiers = ['AC', 'R.N.', 'N-SL-AC', 'Panal', 'High Way']
                for match in days_matches:
                    match_clean = match.strip()
                    # Check if it's a day pattern (contains day names or Odd/Even)
                    if any(day in match_clean.lower() for day in ['day', 'friday', 'saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'odd', 'even', 'normal']):
                        days = match_clean
                        break
                    # Check if it's NOT a route identifier
                    elif not any(identifier.lower() in match_clean.lower() for identifier in route_identifiers):
                        # If it doesn't look like a route identifier, use it
                        if len(match_clean) > 2:  # Avoid single letters/numbers
                            days = match_clean
                            break
        
        return route_type, days
    
    def is_location_header(self, header: str) -> bool:
        """Check if header is a location (not metadata like 'Trip Number')"""
        location_keywords = ['trip', 'number', 'bus running', 'running no', 'no', 'pair', 'route type', 'bus number']
        header_lower = header.lower().strip()
        return not any(keyword in header_lower for keyword in location_keywords)
    
    def normalize_location_name(self, location: str) -> str:
        """Normalize location name by removing Arrival, Departure, Time suffixes"""
        # Remove common suffixes
        normalized = location.strip()
        # Remove suffixes like " Arrival", " Departure", " Time"
        for suffix in [' Arrival', ' Departure', ' Time']:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break
        return normalized.strip()
    
    def extract_locations_from_headers(self, headers: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Extract location names from headers and map original headers to normalized names
        Returns: (normalized_locations, header_to_normalized_map)
        """
        locations = []
        header_to_normalized = {}
        
        for header in headers:
            if self.is_location_header(header):
                normalized = self.normalize_location_name(header)
                if normalized not in locations:
                    locations.append(normalized)
                header_to_normalized[header.strip()] = normalized
        
        return locations, header_to_normalized
    
    def parse_time(self, time_str: str) -> Optional[str]:
        """Parse and validate time string"""
        if time_str is None:
            return None
        if isinstance(time_str, float) and pd.isna(time_str):
            return None
        time_str = str(time_str).strip()
        if not time_str or time_str == '' or time_str.lower() == 'nan':
            return None
        # Handle formats like "4:00", "04:00", "16:00"
        if re.match(r'^\d{1,2}:\d{2}$', time_str):
            return time_str
        return None
    
    def create_location_nodes(self, session, locations: List[str]):
        """Create Location nodes for unique locations"""
        query = """
        UNWIND $locations AS loc_name
        MERGE (l:Location {name: loc_name})
        RETURN count(l) as created
        """
        result = session.run(query, locations=locations)
        return result.single()["created"]
    
    def create_trip_relationship(self, session, from_loc: str, to_loc: str, 
                                  dep_time: str, arr_time: str, 
                                  route_type: Optional[str] = None,
                                  days: Optional[str] = None):
        """Create a TRIP relationship between locations"""
        # Check if relationship with same properties already exists
        # Include route_type and days in the check to allow same trip with different route types/days
        check_query = """
        MATCH (from:Location {name: $from_loc})-[t:TRIP]->(to:Location {name: $to_loc})
        WHERE t.departure_time = $dep_time 
          AND t.arrival_time = $arr_time
          AND COALESCE(t.route_type, '') = COALESCE($route_type, '')
          AND COALESCE(t.days, '') = COALESCE($days, '')
        RETURN t LIMIT 1
        """
        result = session.run(check_query,
                            from_loc=from_loc,
                            to_loc=to_loc,
                            dep_time=dep_time,
                            arr_time=arr_time,
                            route_type=route_type,
                            days=days)
        
        if result.single():
            # Relationship already exists, skip
            return False
        
        # Create new relationship
        create_query = """
        MATCH (from:Location {name: $from_loc})
        MATCH (to:Location {name: $to_loc})
        CREATE (from)-[t:TRIP {
            departure_time: $dep_time,
            arrival_time: $arr_time,
            route_type: $route_type,
            days: $days,
            created_at: timestamp()
        }]->(to)
        RETURN t
        """
        try:
            result = session.run(create_query, 
                                from_loc=from_loc,
                                to_loc=to_loc,
                                dep_time=dep_time,
                                arr_time=arr_time,
                                route_type=route_type,
                                days=days)
            return result.single() is not None
        except Exception as e:
            print(f"    ⚠️  Warning creating relationship: {e}")
            return False
    
    def process_table(self, session, table: Dict, route_type: Optional[str], 
                     days: Optional[str], file_path: str) -> Tuple[int, int]:
        """
        Process a single table and create nodes/relationships
        Returns: (locations_created, trips_created)
        """
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        table_number = table.get("table_number", 1)
        
        # Extract normalized locations from headers
        locations, header_to_normalized = self.extract_locations_from_headers(headers)
        if len(locations) < 2:
            print(f"  ⚠️  Table {table_number}: Not enough locations in headers: {headers}")
            return 0, 0
        
        # Create location nodes with normalized names
        locations_created = self.create_location_nodes(session, locations)
        
        # Build header to index mapping
        header_to_idx = {header.strip(): idx for idx, header in enumerate(headers)}
        
        # Process rows - create relationships between consecutive locations
        trips_created = 0
        
        for row in rows:
            if not row or len(row) < 2:
                continue
            
            # For each pair of consecutive locations
            for i in range(len(locations) - 1):
                from_loc_normalized = locations[i]
                to_loc_normalized = locations[i + 1]
                
                # Find original headers that map to these normalized locations
                from_header = None
                to_header = None
                
                for orig_header, normalized in header_to_normalized.items():
                    if normalized == from_loc_normalized and from_header is None:
                        from_header = orig_header
                    if normalized == to_loc_normalized and to_header is None:
                        to_header = orig_header
                
                if not from_header or not to_header:
                    continue
                
                # Get column indices
                from_idx = header_to_idx.get(from_header)
                to_idx = header_to_idx.get(to_header)
                
                if from_idx is None or to_idx is None:
                    continue
                
                if from_idx >= len(row) or to_idx >= len(row):
                    continue
                
                # Parse times - determine if they are arrival or departure
                from_time = self.parse_time(row[from_idx])
                to_time = self.parse_time(row[to_idx])
                
                # Determine departure and arrival times based on header names
                dep_time = None
                arr_time = None
                
                if from_time and to_time:
                    # Check header names to determine what the times represent
                    from_header_lower = from_header.lower()
                    to_header_lower = to_header.lower()
                    
                    # Case 1: From location has "Departure" -> it's departure from that location
                    if 'departure' in from_header_lower:
                        dep_time = from_time
                        # To location should have "Arrival" or "Time" -> it's arrival at that location
                        if 'arrival' in to_header_lower or 'time' in to_header_lower:
                            arr_time = to_time
                        else:
                            arr_time = to_time  # Default
                    
                    # Case 2: To location has "Arrival" or "Time" -> it's arrival at that location
                    elif 'arrival' in to_header_lower or 'time' in to_header_lower:
                        arr_time = to_time
                        # From location might have departure, or it's a location name with time
                        if 'departure' in from_header_lower:
                            dep_time = from_time
                        elif 'arrival' in from_header_lower:
                            # Both arrivals - this represents time at from location to time at to location
                            # For this case, use from_time as departure and to_time as arrival
                            dep_time = from_time
                        else:
                            # From is just location name, treat as departure
                            dep_time = from_time
                    
                    # Case 3: Default - treat as consecutive locations with times
                    else:
                        dep_time = from_time
                        arr_time = to_time
                    
                    if dep_time and arr_time:
                        if self.create_trip_relationship(session,
                                                         from_loc_normalized,
                                                         to_loc_normalized,
                                                         dep_time,
                                                         arr_time,
                                                         route_type,
                                                         days):
                            trips_created += 1
        
        return locations_created, trips_created
    
    def ingest_json_file(self, json_file_path: str) -> Dict:
        """
        Ingest a single JSON file into Neo4j
        Returns statistics about the ingestion
        """
        print(f"\n📄 Processing: {json_file_path}")
        
        # Load JSON file
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ❌ Error loading JSON: {e}")
            return {"success": False, "error": str(e)}
        
        if not data.get("success") or "extracted_data" not in data:
            print(f"  ⚠️  Invalid JSON structure")
            return {"success": False, "error": "Invalid JSON structure"}
        
        # Extract route info from file path
        route_type, days = self.extract_route_info(json_file_path)
        print(f"  📍 Route Type: {route_type or 'N/A'}")
        print(f"  📅 Days: {days or 'All Days'}")
        
        tables = data.get("extracted_data", {}).get("tables", [])
        print(f"  📊 Found {len(tables)} table(s)")
        
        total_locations = 0
        total_trips = 0
        
        with self.driver.session(database=self.database) as session:
            for table in tables:
                locs, trips = self.process_table(session, table, route_type, days, json_file_path)
                total_locations += locs
                total_trips += trips
        
        print(f"  ✅ Created {total_locations} locations, {total_trips} trips")
        
        return {
            "success": True,
            "file": json_file_path,
            "route_type": route_type,
            "days": days,
            "tables_processed": len(tables),
            "locations_created": total_locations,
            "trips_created": total_trips
        }
    
    def verify_ingestion(self) -> Dict:
        """Verify the ingestion by querying the database"""
        print("\n🔍 Verifying ingestion...")
        
        with self.driver.session(database=self.database) as session:
            # Count locations
            result = session.run("MATCH (l:Location) RETURN count(l) as count")
            location_count = result.single()["count"]
            
            # Count trips
            result = session.run("MATCH ()-[t:TRIP]->() RETURN count(t) as count")
            trip_count = result.single()["count"]
            
            # Count unique route types
            result = session.run("""
                MATCH ()-[t:TRIP]->()
                WHERE t.route_type IS NOT NULL
                RETURN collect(DISTINCT t.route_type) as route_types
            """)
            route_types = result.single()["route_types"]
            
            # Sample trips
            result = session.run("""
                MATCH (from:Location)-[t:TRIP]->(to:Location)
                RETURN from.name as from_loc, to.name as to_loc, 
                       t.departure_time as dep_time, t.arrival_time as arr_time,
                       t.route_type as route_type, t.days as days
                LIMIT 10
            """)
            sample_trips = [dict(record) for record in result]
        
        print(f"  📊 Total Locations: {location_count}")
        print(f"  📊 Total Trips: {trip_count}")
        print(f"  📊 Route Types: {route_types}")
        
        print(f"\n  📋 Sample Trips:")
        for trip in sample_trips:
            route_info = f" ({trip['route_type']})" if trip['route_type'] else ""
            days_info = f" [{trip['days']}]" if trip['days'] else ""
            print(f"    • {trip['from_loc']} -> {trip['to_loc']}: {trip['dep_time']} - {trip['arr_time']}{route_info}{days_info}")
        
        return {
            "location_count": location_count,
            "trip_count": trip_count,
            "route_types": route_types,
            "sample_trips": sample_trips
        }


def main():
    """Main function to test ingestion with a sample file"""
    import sys
    
    # Initialize ingester
    ingester = TimeScheduleIngester()
    
    if not ingester.connect():
        print("❌ Failed to connect to Neo4j")
        return
    
    try:
        # Test with sample file
        sample_file = r"ntc_time_schedule\Extracted\Done-verified\Expressway\Ambalangoda - Colombo_extracted_tables_english.json"
        
        if not os.path.exists(sample_file):
            print(f"❌ Sample file not found: {sample_file}")
            print("Please provide a valid JSON file path")
            if len(sys.argv) > 1:
                sample_file = sys.argv[1]
            else:
                return
        
        # Ingest the file
        result = ingester.ingest_json_file(sample_file)
        
        if result["success"]:
            print(f"\n✅ Successfully ingested {sample_file}")
        else:
            print(f"\n❌ Failed to ingest: {result.get('error', 'Unknown error')}")
        
        # Verify ingestion
        ingester.verify_ingestion()
        
    finally:
        ingester.close()


if __name__ == "__main__":
    main()

