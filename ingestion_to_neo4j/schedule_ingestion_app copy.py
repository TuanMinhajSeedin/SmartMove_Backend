#!/usr/bin/env python3
"""
Flask Application for Schedule Data Ingestion to Neo4j
Allows uploading JSON schedule files and ingesting them into Neo4j database
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from neo4j import GraphDatabase
import json
import os
import re
from typing import List, Dict, Any
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['SECRET_KEY'] = 'schedule-ingestion-secret-key'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "20665130@mM"

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def extract_route_type_from_path(pdf_path: str) -> str:
    """
    Extract route type from PDF path
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Route type string (e.g., "Expressway", "Luxury", "Semi_Luxury", "Normal")
    """
    if not pdf_path:
        return "Unknown"
    
    # Normalize path separators
    path = pdf_path.replace('\\', '/')
    path_lower = path.lower()
    
    # Check for Expressway
    if 'expressway' in path_lower:
        return "Expressway"
    
    # Check for Semi Luxury (check before Luxury to avoid false matches)
    if 'semi luxury' in path_lower or 'semi_luxury' in path_lower:
        return "Semi_Luxury"
    
    # Check for Luxury
    if 'luxury' in path_lower:
        return "Luxury"
    
    # Check for Normal
    if 'normal' in path_lower:
        return "Normal"
    
    # Default
    return "Unknown"


def parse_schedule_json(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse the schedule JSON file and extract table data
    
    Args:
        json_data: The parsed JSON data from the uploaded file
        
    Returns:
        List of parsed table data with headers and rows
    """
    parsed_tables = []
    
    if 'extracted_data' in json_data and 'tables' in json_data['extracted_data']:
        tables = json_data['extracted_data']['tables']
        
        for table in tables:
            headers = table.get('headers', [])
            rows = table.get('rows', [])
            table_number = table.get('table_number', 0)
            
            # Filter headers to only include locations (skip metadata columns)
            # Non-location columns to ignore
            non_location_keywords = [
                'trip', 'number', 'no', 'bus', 'running', 'id', 'code',
                'route', 'service', 'type', 'class', 'status', 'note', 'remark'
            ]
            
            location_headers = []
            metadata_indices = []  # Track indices of non-location columns
            
            for idx, header in enumerate(headers):
                # Clean header name: replace newlines with spaces and strip
                cleaned_header = header.replace('\n', ' ').replace('\r', ' ').strip()
                # Normalize multiple spaces to single space
                cleaned_header = re.sub(r'\s+', ' ', cleaned_header)
                header_lower = cleaned_header.lower()
                
                # Check if header matches any non-location keyword
                is_metadata = any(keyword in header_lower for keyword in non_location_keywords)
                
                if is_metadata:
                    metadata_indices.append(idx)
                else:
                    # Verify it's a location by checking if values look like times
                    # Sample rows to see if values are time-like (ignore empty values)
                    is_location = False
                    sample_count = min(10, len(rows))  # Sample more rows
                    time_like_count = 0
                    non_empty_count = 0
                    
                    for row in rows[:sample_count]:
                        if idx < len(row):
                            value = str(row[idx]).strip()
                            # Skip empty values in validation
                            if value and value != '':
                                non_empty_count += 1
                                # Check if value looks like time (HH:MM or H:MM format)
                                if re.match(r'^\d{1,2}:\d{2}', value):
                                    time_like_count += 1
                    
                    # If we have non-empty values and most of them look like times, it's a location column
                    # Also accept if header contains location keywords (departure, arrival, etc.)
                    location_keywords = ['departure', 'arrival', 'time', 'schedule']
                    has_location_keyword = any(kw in header_lower for kw in location_keywords)
                    
                    if non_empty_count > 0:
                        # If at least 50% of non-empty values are time-like, it's a location
                        time_ratio = time_like_count / non_empty_count if non_empty_count > 0 else 0
                        # Be more lenient for headers with location keywords
                        if time_ratio >= 0.5:
                            is_location = True
                        elif has_location_keyword and time_ratio >= 0.2:  # Lower threshold for location keywords
                            is_location = True
                        elif has_location_keyword and time_like_count > 0:  # If it has any time values and location keyword
                            is_location = True
                    elif has_location_keyword:
                        # If header has location keywords, always consider it a location (even if all empty)
                        is_location = True
                    elif time_like_count > 0:
                        # If we found any time-like values, it's likely a location
                        is_location = True
                    
                    if is_location:
                        # Extract location name and time type from header
                        # Patterns: "Location Arrival", "Location Departure", "Location Time"
                        location_name = None
                        time_type = None
                        
                        # Check for arrival/departure/time patterns
                        if 'arrival' in header_lower:
                            # Extract location name (everything before "Arrival")
                            location_name = re.sub(r'\s+arrival.*$', '', cleaned_header, flags=re.IGNORECASE).strip()
                            time_type = 'arrival'
                        elif 'departure' in header_lower:
                            # Extract location name (everything before "Departure")
                            location_name = re.sub(r'\s+departure.*$', '', cleaned_header, flags=re.IGNORECASE).strip()
                            time_type = 'departure'
                        elif 'time' in header_lower and not any(kw in header_lower for kw in ['arrival', 'departure']):
                            # Extract location name (everything before "Time")
                            location_name = re.sub(r'\s+time.*$', '', cleaned_header, flags=re.IGNORECASE).strip()
                            time_type = 'time'  # Treat as arrival time
                        else:
                            # If no pattern matches, use the full header as location name
                            location_name = cleaned_header
                            time_type = 'time'
                        
                        location_headers.append({
                            'name': location_name,  # Extracted location name
                            'original_header': cleaned_header,  # Keep original for display
                            'time_type': time_type,  # arrival, departure, or time
                            'index': idx
                        })
                    else:
                        metadata_indices.append(idx)
            
            # Parse rows - extract location times (ignore metadata columns)
            parsed_rows = []
            for row in rows:
                if len(row) != len(headers):
                    continue
                
                # Extract trip/bus number from first metadata column if available
                trip_number = None
                if metadata_indices and len(metadata_indices) > 0:
                    trip_number = row[metadata_indices[0]] if metadata_indices[0] < len(row) else None
                
                # Extract times for each location with time type information
                location_times = {}
                location_data = {}  # {location_name: {arrival: time, departure: time, time: time}}
                
                for loc_header in location_headers:
                    loc_name = loc_header['name']
                    original_header = loc_header.get('original_header', loc_name)
                    time_type = loc_header.get('time_type', 'time')
                    loc_index = loc_header['index']
                    
                    if loc_index < len(row):
                        time_value = str(row[loc_index]).strip() if row[loc_index] is not None else ""
                        if time_value:
                            # Store in location_times using original header for backward compatibility
                            location_times[original_header] = time_value
                            
                            # Store in location_data grouped by location name
                            if loc_name not in location_data:
                                location_data[loc_name] = {}
                            location_data[loc_name][time_type] = time_value
                        else:
                            location_times[original_header] = None
                
                parsed_rows.append({
                    'trip_number': trip_number,
                    'location_times': location_times,  # Keep for display (uses original headers)
                    'location_data': location_data  # Grouped by location name with time types
                })
            
            parsed_tables.append({
                'table_number': table_number,
                'headers': [h.get('original_header', h['name']) for h in location_headers],  # Use original for display
                'location_headers': location_headers,  # Store full info for ingestion
                'rows': parsed_rows,
                'page_number': table.get('page_number', 1)
            })
    
    return parsed_tables


def ingest_schedule_to_neo4j(parsed_tables: List[Dict[str, Any]], route_name: str = None, route_type: str = None):
    """
    Ingest schedule data into Neo4j
    Creates only Location nodes and SCHEDULE relationships between consecutive locations
    
    Args:
        parsed_tables: List of parsed table data
        route_name: Optional route name for grouping schedules
        route_type: Route type (e.g., "Expressway", "Luxury", "Semi_Luxury", "Normal")
    """
    with driver.session() as session:
        for table in parsed_tables:
            location_headers_info = table.get('location_headers', [])
            rows = table['rows']
            
            # Process each row to create schedule relationships
            for row in rows:
                trip_number = row.get('trip_number')
                location_data = row.get('location_data', {})  # Grouped by location name
                
                # Build ordered list of unique locations based on header order
                locations_ordered = []
                seen_locations = set()
                
                for loc_info in location_headers_info:
                    loc_name = loc_info['name']
                    if loc_name in location_data and loc_name not in seen_locations:
                        # Check if location has any valid time values
                        if any(time_val for time_val in location_data[loc_name].values() if time_val):
                            locations_ordered.append(loc_name)
                            seen_locations.add(loc_name)
                
                # Create schedule relationships between consecutive locations
                for i in range(len(locations_ordered) - 1):
                    from_location = locations_ordered[i]
                    to_location = locations_ordered[i + 1]
                    
                    from_data = location_data.get(from_location, {})
                    to_data = location_data.get(to_location, {})
                    
                    # Get times: use departure from source, arrival at destination
                    from_time = from_data.get('departure') or from_data.get('time') or (list(from_data.values())[0] if from_data else None)
                    to_time = to_data.get('arrival') or to_data.get('time') or (list(to_data.values())[0] if to_data else None)
                    
                    # Skip if either time is missing or empty (both must be present)
                    if not from_time or not to_time or from_time == "" or to_time == "":
                        continue
                    
                    # Create or update location nodes (only Location nodes, no Trip or Route)
                    session.run("""
                        MERGE (from:Location {name: $from_location})
                        MERGE (to:Location {name: $to_location})
                    """, from_location=from_location, to_location=to_location)
                    
                    # Create schedule relationship with trip information
                    schedule_props = {
                        'trip_number': trip_number,
                        'from_time': from_time,
                        'to_time': to_time,
                        'table_number': table['table_number']
                    }
                    
                    # Add arrival/departure properties
                    # Arrival at the source location (from_location)
                    if 'arrival' in from_data:
                        schedule_props['arrival_to_terminal'] = from_data['arrival']
                    # Departure from the source location (from_location)
                    if 'departure' in from_data:
                        schedule_props['departure_from_terminal'] = from_data['departure']
                    
                    if route_name:
                        schedule_props['route_name'] = route_name
                    
                    if route_type:
                        schedule_props['route_type'] = route_type
                    
                    # Create SCHEDULE relationship (handles both directions automatically)
                    session.run("""
                        MATCH (from:Location {name: $from_location})
                        MATCH (to:Location {name: $to_location})
                        CREATE (from)-[s:SCHEDULE]->(to)
                        SET s += $props
                    """, from_location=from_location, to_location=to_location, props=schedule_props)


@app.route('/')
def index():
    """Main page"""
    return render_template('schedule_ingestion.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and parse JSON"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
        
        if not file.filename.endswith('.json'):
            return jsonify({
                'success': False,
                'message': 'Please upload a JSON file'
            }), 400
        
        # Read and parse JSON
        json_data = json.load(file)
        
        # Parse the schedule data
        parsed_tables = parse_schedule_json(json_data)
        
        if not parsed_tables:
            return jsonify({
                'success': False,
                'message': 'No valid schedule tables found in the file'
            }), 400
        
        # Extract route name and route type from filename or PDF path if available
        route_name = None
        route_type = None
        if 'pdf_path' in json_data:
            # Extract route name and type from PDF path
            pdf_path = json_data['pdf_path']
            route_name = os.path.basename(pdf_path).replace('.pdf', '').strip()
            route_type = extract_route_type_from_path(pdf_path)
        else:
            route_name = file.filename.replace('.json', '').replace('_extracted_tables_english', '').strip()
            route_type = "Unknown"
        
        return jsonify({
            'success': True,
            'parsed_tables': parsed_tables,
            'route_name': route_name,
            'route_type': route_type,
            'message': f'Successfully parsed {len(parsed_tables)} table(s)'
        })
        
    except json.JSONDecodeError:
        return jsonify({
            'success': False,
            'message': 'Invalid JSON file'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing file: {str(e)}'
        }), 500


@app.route('/api/ingest', methods=['POST'])
def ingest_data():
    """Ingest parsed schedule data into Neo4j"""
    try:
        data = request.get_json()
        
        if 'parsed_tables' not in data:
            return jsonify({
                'success': False,
                'message': 'No parsed tables provided'
            }), 400
        
        parsed_tables = data['parsed_tables']
        route_name = data.get('route_name', 'Unknown')
        route_type = data.get('route_type', 'Unknown')
        
        # Test Neo4j connection
        try:
            with driver.session() as session:
                session.run("RETURN 1")
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Cannot connect to Neo4j: {str(e)}'
            }), 500
        
        # Ingest data
        ingest_schedule_to_neo4j(parsed_tables, route_name, route_type)
        
        # Get statistics
        with driver.session() as session:
            # Count locations
            result = session.run("MATCH (l:Location) RETURN count(l) as count")
            location_count = result.single()['count']
            
            # Count schedules
            result = session.run("MATCH ()-[s:SCHEDULE]->() RETURN count(s) as count")
            schedule_count = result.single()['count']
        
        return jsonify({
            'success': True,
            'message': 'Data successfully ingested into Neo4j',
            'statistics': {
                'locations': location_count,
                'schedules': schedule_count
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error ingesting data: {str(e)}'
        }), 500


@app.route('/api/test-connection', methods=['GET'])
def test_connection():
    """Test Neo4j connection"""
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()['test']
            
            # Get database stats
            result = session.run("MATCH (l:Location) RETURN count(l) as location_count")
            location_count = result.single()['location_count']
            
            result = session.run("MATCH ()-[s:SCHEDULE]->() RETURN count(s) as schedule_count")
            schedule_count = result.single()['schedule_count']
            
            return jsonify({
                'success': True,
                'connected': True,
                'statistics': {
                    'locations': location_count,
                    'schedules': schedule_count
                }
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'connected': False,
            'message': str(e)
        }), 500


@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Get list of all unique location names from Neo4j"""
    try:
        with driver.session() as session:
            # Get all unique location names, sorted alphabetically
            result = session.run("""
                MATCH (l:Location)
                RETURN DISTINCT l.name as location_name
                ORDER BY l.name ASC
            """)
            
            locations = [record['location_name'] for record in result]
            
            return jsonify({
                'success': True,
                'locations': locations,
                'count': len(locations)
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error fetching locations: {str(e)}',
            'locations': []
        }), 500


if __name__ == '__main__':
    print("🚀 Starting Schedule Ingestion App...")
    print(f"📊 Neo4j URI: {NEO4J_URI}")
    print(f"🌐 Server running on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)

