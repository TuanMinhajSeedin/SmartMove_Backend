# Schedule Data Ingestion App

A Flask web application for uploading and ingesting schedule JSON files into Neo4j database.

## Features

- 📁 **File Upload**: Upload JSON schedule files extracted from PDFs
- 📊 **Data Preview**: View parsed schedule data with headers and rows before ingestion
- 🔗 **Neo4j Integration**: Connect to Neo4j schedule database and ingest data
- 📍 **Location Mapping**: Automatically maps locations in the same order as headers
- 🚀 **Batch Ingestion**: Efficiently ingests schedule data with relationships

## Requirements

- Python 3.7+
- Neo4j database running (default: `bolt://localhost:7687`)
- Neo4j credentials: username `neo4j`, password `20665130@mM`

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure Neo4j is running and accessible at `bolt://localhost:7687`

3. Update Neo4j credentials in `schedule_ingestion_app.py` if needed:
```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "20665130@mM"
```

## Usage

1. Start the Flask application:
```bash
python schedule_ingestion_app.py
```

2. Open your browser and navigate to:
```
http://localhost:5001
```

3. **Upload JSON File**:
   - Click "Choose File" and select a JSON schedule file
   - Click "Upload & Parse" to parse the file
   - The parsed data will be displayed in tables showing:
     - Headers (location names)
     - Rows (trip numbers and times for each location)

4. **Review Data**:
   - Check the parsed tables to ensure data is correct
   - Verify that locations are mapped in the correct order

5. **Ingest to Neo4j**:
   - Click "🚀 Ingest Data into Neo4j" button
   - Wait for the ingestion to complete
   - View statistics showing number of locations, schedules, and trips ingested

## JSON File Format

The app expects JSON files with the following structure:

```json
{
  "success": true,
  "pdf_path": "...",
  "extracted_data": {
    "tables": [
      {
        "table_number": 1,
        "headers": ["Trip Number", "Location1", "Location2", "Location3"],
        "rows": [
          ["Trip 01", "4:00", "6:00", "8:00"],
          ["Trip 02", "5:00", "7:00", "9:00"]
        ],
        "page_number": 1
      }
    ]
  }
}
```

## Neo4j Data Model

The app creates the following structure in Neo4j:

### Nodes:
- **Location**: Represents a bus stop/location
  - Properties: `name`
- **Route**: Represents a bus route
  - Properties: `name`, `last_updated`
- **Trip**: Represents a specific trip
  - Properties: `trip_id`, `trip_number`, `route_name`, `table_number`, `time_*` (for each location)

### Relationships:
- **SCHEDULE**: Between consecutive locations
  - Properties: `trip_number`, `from_time`, `to_time`, `table_number`, `route_name`
- **STARTS_AT**: From Trip to first Location
- **STOPS_AT**: From Trip to intermediate Locations
  - Properties: `stop_order`

## API Endpoints

- `GET /` - Main page with file upload interface
- `POST /api/upload` - Upload and parse JSON file
- `POST /api/ingest` - Ingest parsed data into Neo4j
- `GET /api/test-connection` - Test Neo4j connection

## Example

1. Upload `Ambalangoda - Colombo_extracted_tables_english.json`
2. The app will parse:
   - Table 1: Ambalangoda → Colombo (18 trips)
   - Table 2: Colombo → Makumbura → Ambalangoda (18 trips)
3. After ingestion, Neo4j will contain:
   - Location nodes: Ambalangoda, Colombo, Makumbura
   - Schedule relationships between consecutive locations
   - Trip nodes with all stop times

## Troubleshooting

- **Connection Error**: Ensure Neo4j is running and credentials are correct
- **File Parse Error**: Check that JSON file matches expected format
- **Ingestion Error**: Verify Neo4j database is accessible and has sufficient space

## Notes

- Locations are mapped in the same order as they appear in headers
- Each table is processed separately
- Route name is extracted from PDF filename or JSON metadata
- Duplicate trips are handled with unique trip IDs


