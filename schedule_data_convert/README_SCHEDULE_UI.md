# Schedule Data Ingestion UI

A Streamlit web application for safely ingesting schedule data into Neo4j Aura instances without destroying existing data.

## Features

✅ **Safe Data Ingestion** - Uses MERGE operations to preserve existing data  
✅ **File Upload** - Upload JSON files with schedule data  
✅ **Automatic Format Conversion** - Converts various JSON formats to standard schedule format  
✅ **Dynamic Properties** - Add and edit optional properties via UI  
✅ **Manual Data Entry** - Enter schedules manually  
✅ **Real-time Database Stats** - View current database statistics  
✅ **Place Node Management** - Automatically creates Place nodes if they don't exist  
✅ **Schedule Relationships** - Creates Schedule relationships with departure/arrival times  

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the Application

```bash
streamlit run schedule_ingestion_ui.py
```

### Steps to Ingest Data

1. **Connect to Neo4j**
   - Enter your Neo4j Aura connection details in the sidebar
   - Click "Connect to Neo4j"
   - Verify connection status

2. **Add Optional Properties (Optional)**
   - Add properties like `route_type`, `service_class`, etc.
   - These will be added to all Schedule relationships
   - Edit or delete properties as needed

3. **Upload JSON File**
   - Click "Choose a JSON file"
   - Select a JSON file containing schedule data
   - The app will automatically convert it to the required format

4. **Review Data**
   - Check the converted data preview
   - Verify schedules are correct

5. **Ingest Data**
   - Click "Ingest Data to Neo4j"
   - Watch progress bar
   - View results

### JSON File Format

The app accepts multiple JSON formats and converts them to:

```json
{
  "from": "Colombo",
  "to": "Kandy",
  "schedules": [
    {"departure": "06:00", "arrival": "09:30"},
    {"departure": "08:00", "arrival": "11:30"},
    {"departure": "14:00", "arrival": "17:30"}
  ]
}
```

**Supported Input Formats:**
- Vision extractor format (with `extracted_data.tables`)
- Direct format (with `from`, `to`, `schedules`)
- List of route objects

### Manual Data Entry

You can also enter schedules manually:
1. Fill in "From Place" and "To Place"
2. Add schedules with departure and arrival times (HH:MM format)
3. Click "Ingest Manual Data"

### Optional Properties

Add properties that will be included in all Schedule relationships:
- Examples: `route_type`, `service_class`, `valid_from`, etc.
- All properties are editable via text boxes
- Properties are applied to all schedules created in that session

## Neo4j Schema

### Nodes

**Place**
```cypher
(:Place {name: String})
```

### Relationships

**Schedule**
```cypher
(:Place)-[:Schedule {
  departure: String,
  arrival: String,
  ...optional_properties
}]->(:Place)
```

## Safety Features

- ✅ Uses `MERGE` for Place nodes (creates only if doesn't exist)
- ✅ Uses `CREATE` for Schedule relationships (allows duplicates with same times)
- ✅ Never deletes existing data
- ✅ Connection validation before operations
- ✅ Error handling and reporting

## Environment Variables

You can set these environment variables to pre-fill connection details:

```bash
export NEO4J_URI="neo4j+ssc://your-instance.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
export NEO4J_DATABASE="neo4j"
```

## Example Workflow

1. **First Time Setup**
   ```bash
   streamlit run schedule_ingestion_ui.py
   ```

2. **Connect to Database**
   - Enter Aura credentials
   - Click "Connect"

3. **Add Route Type Property**
   - Key: `route_type`
   - Value: `Expressway`
   - Click "Add Property"

4. **Upload JSON File**
   - Select your schedule JSON file
   - Review converted data

5. **Ingest**
   - Click "Ingest Data to Neo4j"
   - Wait for completion
   - Check results

## Troubleshooting

**Connection Issues:**
- Verify URI format: `neo4j+ssc://instance.databases.neo4j.io`
- Check username and password
- Ensure database name is correct

**Data Not Showing:**
- Check JSON file format
- Verify file contains schedule data
- Check browser console for errors

**Properties Not Applied:**
- Ensure properties are added before ingestion
- Properties are shown in info box before ingestion
- Check that properties don't contain special characters

## Notes

- All Place nodes are created using MERGE (safe, won't duplicate)
- Schedule relationships are created with CREATE (allows multiple schedules between same places)
- Time format should be HH:MM (e.g., "06:00", "14:30")
- The app preserves all existing data in your database


