#!/usr/bin/env python3
"""
Streamlit UI for ingesting schedule data into Neo4j Aura instance
- Safe ingestion (preserves existing data)
- Creates Place nodes if they don't exist
- Creates Schedule relationships with properties
- File upload for JSON files
- Dynamic UI for optional properties
"""

import streamlit as st
import json
import os
from neo4j import GraphDatabase
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Schedule Data Ingestion",
    page_icon="🚌",
    layout="wide"
)

st.title("🚌 Schedule Data Ingestion to Neo4j")
st.markdown("Safely ingest schedule data into your Neo4j Aura instance (preserves existing data)")

# Initialize session state
if 'neo4j_connected' not in st.session_state:
    st.session_state.neo4j_connected = False
if 'driver' not in st.session_state:
    st.session_state.driver = None
if 'optional_properties' not in st.session_state:
    st.session_state.optional_properties = {}

# Sidebar for Neo4j connection
with st.sidebar:
    st.header("🔌 Neo4j Connection")
    
    # Connection settings
    neo4j_uri = st.text_input(
        "Neo4j URI",
        value=os.getenv("NEO4J_URI", "neo4j+ssc://10e45f8e.databases.neo4j.io"),
        help="Neo4j Aura connection URI"
    )
    
    neo4j_user = st.text_input(
        "Username",
        value=os.getenv("NEO4J_USER", "neo4j"),
        type="default"
    )
    
    neo4j_password = st.text_input(
        "Password",
        value=os.getenv("NEO4J_PASSWORD", ""),
        type="password",
        help="Enter your Neo4j password"
    )
    
    neo4j_database = st.text_input(
        "Database",
        value=os.getenv("NEO4J_DATABASE", "neo4j"),
        help="Database name (default: neo4j)"
    )
    
    # Connect button
    if st.button("🔗 Connect to Neo4j", type="primary", use_container_width=True):
        try:
            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            # Test connection
            with driver.session(database=neo4j_database) as session:
                result = session.run("RETURN 1 as test")
                if result.single()["test"] == 1:
                    st.session_state.driver = driver
                    st.session_state.neo4j_connected = True
                    st.session_state.neo4j_database = neo4j_database
                    st.success("✅ Connected to Neo4j!")
                else:
                    st.error("❌ Connection test failed")
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")
            st.session_state.neo4j_connected = False
    
    # Disconnect button
    if st.session_state.neo4j_connected:
        if st.button("🔌 Disconnect", use_container_width=True):
            if st.session_state.driver:
                st.session_state.driver.close()
            st.session_state.driver = None
            st.session_state.neo4j_connected = False
            st.success("Disconnected")
            st.rerun()
    
    # Show connection status
    if st.session_state.neo4j_connected:
        st.info("🟢 Connected")
        
        # Show database stats
        try:
            with st.session_state.driver.session(database=neo4j_database) as session:
                # Count places
                result = session.run("MATCH (p:Place) RETURN count(p) as count")
                place_count = result.single()["count"]
                
                # Count schedules
                result = session.run("MATCH ()-[s:Schedule]->() RETURN count(s) as count")
                schedule_count = result.single()["count"]
                
                st.metric("Places", place_count)
                st.metric("Schedules", schedule_count)
        except Exception as e:
            st.warning(f"Could not fetch stats: {e}")
    else:
        st.warning("🔴 Not Connected")

def convert_json_to_schedule_format(json_data: Dict, optional_properties: Dict = None) -> List[Dict]:
    """
    Convert JSON data to schedule format:
    {
        "from": "Colombo",
        "to": "Kandy",
        "schedules": [
            {"departure": "06:00", "arrival": "09:30", ...optional_properties},
            ...
        ]
    }
    
    Args:
        json_data: Input JSON data
        optional_properties: Dictionary of optional properties to add to each schedule
    """
    if optional_properties is None:
        optional_properties = {}
    
    converted_data = []
    
    # Handle different JSON structures
    if "extracted_data" in json_data:
        # Structure from vision extractor
        tables = json_data.get("extracted_data", {}).get("tables", [])
        
        for table in tables:
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            
            # Extract location columns; ignore arrival-only columns and normalize names
            location_keywords = ['trip', 'number', 'bus', 'running', 'no', 'pair', 'route']
            location_candidates = []
            route_type_index = None

            for idx, header in enumerate(headers):
                if header is None:
                    continue
                header_lower = header.lower().strip()
                if 'route type' in header_lower:
                    route_type_index = idx
                    continue
                if any(keyword in header_lower for keyword in location_keywords):
                    continue

                normalized = header_lower.replace('arrival', '').replace('departure', '').replace('time', '').strip()
                if not normalized:
                    continue
                normalized = normalized.title()
                is_arrival_only = 'arrival' in header_lower and 'departure' not in header_lower
                location_candidates.append((idx, normalized, is_arrival_only))

            # Prefer departure/normal columns over arrival-only columns for the same place
            location_map = {}
            location_order = []
            for idx, normalized, is_arrival_only in location_candidates:
                if normalized not in location_map:
                    location_map[normalized] = (idx, is_arrival_only)
                    location_order.append(normalized)
                elif location_map[normalized][1] and not is_arrival_only:
                    location_map[normalized] = (idx, is_arrival_only)

            locations = []
            location_indices = []
            for normalized in location_order:
                idx, _ = location_map[normalized]
                locations.append(normalized)
                location_indices.append(idx)

            if len(locations) < 2:
                continue
            
            # For each pair of consecutive locations
            for i in range(len(locations) - 1):
                from_loc = locations[i]
                to_loc = locations[i + 1]
                from_idx = location_indices[i]
                to_idx = location_indices[i + 1]
                
                schedules = []
                
                # Extract schedules from rows
                for row in rows:
                    if len(row) > max(from_idx, to_idx):
                        from_time = row[from_idx] if from_idx < len(row) else None
                        to_time = row[to_idx] if to_idx < len(row) else None
                        
                        # Parse times
                        from_time_str = parse_time(from_time)
                        to_time_str = parse_time(to_time)
                        
                        if from_time_str and to_time_str:
                            schedule_item = {
                                "departure": from_time_str,
                                "arrival": to_time_str
                            }
                            if route_type_index is not None and route_type_index < len(row):
                                route_type_value = row[route_type_index]
                                if route_type_value is not None:
                                    route_type_value = str(route_type_value).strip()
                                    if route_type_value and route_type_value.lower() not in ['', 'nan', 'none']:
                                        schedule_item["route_type"] = route_type_value
                            # Add optional properties to each schedule
                            schedule_item.update(optional_properties)
                            schedules.append(schedule_item)
                
                if schedules:
                    converted_data.append({
                        "from": from_loc,
                        "to": to_loc,
                        "schedules": schedules
                    })
    
    elif "from" in json_data and "to" in json_data:
        # Already in correct format - add optional properties to schedules
        if "schedules" in json_data and isinstance(json_data["schedules"], list):
            for schedule in json_data["schedules"]:
                schedule.update(optional_properties)
        converted_data.append(json_data)
    
    elif isinstance(json_data, list):
        # List of schedule objects - add optional properties
        for item in json_data:
            if "schedules" in item and isinstance(item["schedules"], list):
                for schedule in item["schedules"]:
                    schedule.update(optional_properties)
        converted_data = json_data
    
    return converted_data

def parse_time(time_value) -> Optional[str]:
    """Parse time value to HH:MM format"""
    if time_value is None:
        return None
    
    if isinstance(time_value, float) and pd.isna(time_value):
        return None
    
    time_str = str(time_value).strip()
    
    if not time_str or time_str.lower() in ['', 'nan', 'none']:
        return None
    
    # Handle formats like "4:00", "04:00", "16:00", "4.00"
    import re
    
    # Replace dots with colons
    time_str = time_str.replace('.', ':')
    
    # Match HH:MM or H:MM format
    match = re.match(r'^(\d{1,2}):(\d{2})$', time_str)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        
        # Format as HH:MM
        return f"{hours:02d}:{minutes:02d}"
    
    return None

def create_place_node(session, place_name: str) -> bool:
    """Create Place node if it doesn't exist (MERGE - safe)"""
    query = """
    MERGE (p:Place {name: $place_name})
    RETURN p
    """
    try:
        result = session.run(query, place_name=place_name)
        return result.single() is not None
    except Exception as e:
        st.error(f"Error creating place '{place_name}': {e}")
        return False

def create_schedule_relationship(session, from_place: str, to_place: str, 
                                 departure: str, arrival: str,
                                 optional_props: Dict) -> bool:
    """Create Schedule relationship with properties"""
    # Build properties dictionary
    props = {
        'departure': departure,
        'arrival': arrival,
        **optional_props
    }
    
    # Build query dynamically
    prop_strings = [f"{key}: ${key}" for key in props.keys()]
    props_dict = {key: value for key, value in props.items()}
    props_dict['from_place'] = from_place
    props_dict['to_place'] = to_place
    
    query = f"""
    MATCH (from:Place {{name: $from_place}})
    MATCH (to:Place {{name: $to_place}})
    CREATE (from)-[s:Schedule {{{', '.join(prop_strings)}}}]->(to)
    RETURN s
    """
    
    try:
        result = session.run(query, **props_dict)
        return result.single() is not None
    except Exception as e:
        st.error(f"Error creating schedule: {e}")
        return False

# Main content area
if not st.session_state.neo4j_connected:
    st.warning("⚠️ Please connect to Neo4j in the sidebar to continue")
    st.info("💡 Enter your Neo4j Aura connection details and click 'Connect to Neo4j'")
else:
    st.success("✅ Connected to Neo4j - Ready to ingest data")
    
    # Optional Properties Section
    st.header("📝 Optional Properties")
    st.markdown("Add optional properties that will be included in all Schedule relationships")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_prop_key = st.text_input("Property Key", key="new_prop_key", placeholder="e.g., route_type")
    
    with col2:
        new_prop_value = st.text_input("Property Value", key="new_prop_value", placeholder="e.g., Expressway")
    
    col_add, col_clear = st.columns([1, 1])
    
    with col_add:
        if st.button("➕ Add Property", use_container_width=True):
            if new_prop_key and new_prop_value:
                st.session_state.optional_properties[new_prop_key.strip()] = new_prop_value.strip()
                st.success(f"Added: {new_prop_key} = {new_prop_value}")
                st.rerun()
    
    with col_clear:
        if st.button("🗑️ Clear All Properties", use_container_width=True):
            st.session_state.optional_properties = {}
            st.rerun()
    
    # Display and edit existing properties
    if st.session_state.optional_properties:
        st.subheader("Current Optional Properties (Editable)")
        
        properties_to_remove = []
        
        for key, value in st.session_state.optional_properties.items():
            col_key, col_val, col_del = st.columns([2, 3, 1])
            
            with col_key:
                new_key = st.text_input("Key", value=key, key=f"prop_key_{key}")
            
            with col_val:
                new_value = st.text_input("Value", value=value, key=f"prop_val_{key}")
            
            with col_del:
                if st.button("🗑️", key=f"del_{key}", use_container_width=True):
                    properties_to_remove.append(key)
            
            # Update if changed
            if new_key != key or new_value != value:
                if new_key and new_value:
                    del st.session_state.optional_properties[key]
                    st.session_state.optional_properties[new_key] = new_value
                    st.rerun()
        
        # Remove properties
        for key in properties_to_remove:
            del st.session_state.optional_properties[key]
            st.rerun()
    
    # File Upload Section
    st.header("📄 Upload JSON File")
    
    uploaded_file = st.file_uploader(
        "Choose a JSON file",
        type=['json'],
        help="Upload a JSON file containing schedule data",
        key="json_file_uploader"
    )
    
    # Store uploaded file data in session state to allow re-conversion when properties change
    if uploaded_file is not None:
        # Get file identifier
        file_id = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.tell()}"
        
        # If new file or properties changed, reload and convert
        if ('last_file_id' not in st.session_state or 
            st.session_state.last_file_id != file_id or
            'last_properties' not in st.session_state or
            st.session_state.last_properties != st.session_state.optional_properties):
            
            # Read file content
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            
            # Store in session state
            st.session_state.uploaded_file_data = json.loads(file_bytes.decode('utf-8'))
            st.session_state.last_file_id = file_id
            st.session_state.last_properties = st.session_state.optional_properties.copy()
    
    # Process file if we have data
    if 'uploaded_file_data' in st.session_state:
        try:
            json_data = st.session_state.uploaded_file_data
            
            # Convert to schedule format (always include current optional properties)
            with st.spinner("Converting data format..."):
                schedule_data = convert_json_to_schedule_format(json_data, st.session_state.optional_properties)
            
            if schedule_data:
                st.success(f"✅ Converted {len(schedule_data)} route(s)")
                
                # Display converted data with properties
                with st.expander("📋 View Converted Data (with Optional Properties)", expanded=False):
                    if st.session_state.optional_properties:
                        st.info(f"✨ Optional properties included in schedules: {', '.join(st.session_state.optional_properties.keys())}")
                    st.json(schedule_data)
                
                # Show preview table
                st.subheader("📊 Data Preview")
                
                preview_rows = []
                for route in schedule_data:  # Show first 10 routes
                    for schedule in route["schedules"]:  # Show first 3 schedules per route
                        row = {
                            "From": route["from"],
                            "To": route["to"],
                            "Departure": schedule["departure"],
                            "Arrival": schedule["arrival"]
                        }
                        # Add optional properties to preview table
                        for key, value in schedule.items():
                            if key not in ["departure", "arrival"]:
                                row[key] = value
                        preview_rows.append(row)
                
                if preview_rows:
                    df_preview = pd.DataFrame(preview_rows)
                    st.dataframe(df_preview, use_container_width=True)
                    
                    # Show info about properties
                    if st.session_state.optional_properties:
                        st.info(f"📝 Optional properties shown in preview: {', '.join(st.session_state.optional_properties.keys())}")
                
                # Ingestion section
                st.header("💾 Ingest to Neo4j")
                
                # Show summary
                total_schedules = sum(len(route["schedules"]) for route in schedule_data)
                total_routes = len(schedule_data)
                unique_places = set()
                for route in schedule_data:
                    unique_places.add(route["from"])
                    unique_places.add(route["to"])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Routes", total_routes)
                with col2:
                    st.metric("Schedules", total_schedules)
                with col3:
                    st.metric("Places", len(unique_places))
                
                # Show optional properties
                if st.session_state.optional_properties:
                    st.info(f"📝 Optional properties to be added: {st.session_state.optional_properties}")
                
                # Ingest button
                if st.button("🚀 Ingest Data to Neo4j", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    places_created = 0
                    schedules_created = 0
                    errors = []
                    
                    total_items = len(unique_places) + total_schedules
                    processed = 0
                    
                    with st.session_state.driver.session(database=st.session_state.neo4j_database) as session:
                        # Create all places first
                        status_text.text("Creating Place nodes...")
                        for place in unique_places:
                            if create_place_node(session, place):
                                places_created += 1
                            processed += 1
                            progress_bar.progress(processed / total_items)
                        
                        # Create schedule relationships
                        status_text.text("Creating Schedule relationships...")
                        for route in schedule_data:
                            from_place = route["from"]
                            to_place = route["to"]
                            
                            for schedule in route["schedules"]:
                                try:
                                    if create_schedule_relationship(
                                        session,
                                        from_place,
                                        to_place,
                                        schedule["departure"],
                                        schedule["arrival"],
                                        st.session_state.optional_properties
                                    ):
                                        schedules_created += 1
                                except Exception as e:
                                    errors.append(f"{from_place} -> {to_place} ({schedule['departure']}-{schedule['arrival']}): {e}")
                                
                                processed += 1
                                progress_bar.progress(processed / total_items)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Show results
                    st.success(f"✅ Ingestion Complete!")
                    st.metric("Places Created/Merged", places_created)
                    st.metric("Schedules Created", schedules_created)
                    
                    if errors:
                        st.warning(f"⚠️ {len(errors)} errors occurred:")
                        for error in errors[:10]:  # Show first 10 errors
                            st.error(error)
                    
                    # Refresh stats
                    st.rerun()
            
            else:
                st.warning("⚠️ No schedule data found in JSON file")
                st.json(json_data)  # Show raw data for debugging
                
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON file: {e}")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.exception(e)
    
    # Manual data entry section
    st.header("✏️ Manual Data Entry")
    
    col_from, col_to = st.columns(2)
    
    with col_from:
        manual_from = st.text_input("From Place", placeholder="e.g., Colombo")
    
    with col_to:
        manual_to = st.text_input("To Place", placeholder="e.g., Kandy")
    
    # Manual schedules
    st.subheader("Schedules")
    
    if 'manual_schedules' not in st.session_state:
        st.session_state.manual_schedules = [{"departure": "", "arrival": ""}]
    
    for i, schedule in enumerate(st.session_state.manual_schedules):
        col_dep, col_arr, col_del = st.columns([3, 3, 1])
        
        with col_dep:
            st.session_state.manual_schedules[i]["departure"] = st.text_input(
                "Departure", 
                value=schedule["departure"],
                key=f"dep_{i}",
                placeholder="HH:MM"
            )
        
        with col_arr:
            st.session_state.manual_schedules[i]["arrival"] = st.text_input(
                "Arrival",
                value=schedule["arrival"],
                key=f"arr_{i}",
                placeholder="HH:MM"
            )
        
        with col_del:
            if st.button("🗑️", key=f"del_sched_{i}", use_container_width=True):
                st.session_state.manual_schedules.pop(i)
                st.rerun()
    
    if st.button("➕ Add Schedule", use_container_width=True):
        st.session_state.manual_schedules.append({"departure": "", "arrival": ""})
        st.rerun()
    
    if st.button("💾 Ingest Manual Data", type="primary", use_container_width=True):
        if manual_from and manual_to and st.session_state.manual_schedules:
            # Filter out empty schedules
            valid_schedules = [
                s for s in st.session_state.manual_schedules 
                if s["departure"] and s["arrival"]
            ]
            
            if valid_schedules:
                # Parse times
                parsed_schedules = []
                for sched in valid_schedules:
                    dep = parse_time(sched["departure"])
                    arr = parse_time(sched["arrival"])
                    if dep and arr:
                        parsed_schedules.append({"departure": dep, "arrival": arr})
                
                if parsed_schedules:
                    with st.session_state.driver.session(database=st.session_state.neo4j_database) as session:
                        # Create places
                        create_place_node(session, manual_from)
                        create_place_node(session, manual_to)
                        
                        # Create schedules
                        created = 0
                        for sched in parsed_schedules:
                            if create_schedule_relationship(
                                session,
                                manual_from,
                                manual_to,
                                sched["departure"],
                                sched["arrival"],
                                st.session_state.optional_properties
                            ):
                                created += 1
                        
                        st.success(f"✅ Created {created} schedule(s)")
                        st.session_state.manual_schedules = [{"departure": "", "arrival": ""}]
                        st.rerun()
                else:
                    st.error("⚠️ No valid times found. Please use HH:MM format.")
            else:
                st.error("⚠️ Please add at least one schedule")
        else:
            st.error("⚠️ Please fill in From, To, and at least one schedule")

# Cleanup on page unload
if st.session_state.neo4j_connected and st.session_state.driver:
    # Keep connection alive - cleanup handled by disconnect button
    pass

