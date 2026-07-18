#!/usr/bin/env python3
"""
Aligned ingestion script that combines the best features from both ingest.py and complete_all_routes_ingestion.py
- Preserves ALL routes including duplicates (from complete_all_routes_ingestion.py)
- Includes place name normalization and duplicate handling (from ingest.py)
- Uses batch processing for efficiency
"""

import os
import glob
import pandas as pd
from neo4j import GraphDatabase
import time
import difflib

def ingest_all_routes_with_normalization():
    """Ingest ALL routes including duplicates with place name normalization"""
    
    print("🚀 Starting COMPLETE ingestion with place name normalization...")
    
    # Load the original data
    final_tuples = []
    folder = "artifacts/public_transport_data/translated_data/Fares/AC Fares - Katunayake Expressway(Effect from 2024-10-02)"
    csv_files = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    
    print(f"📁 Found {len(csv_files)} CSV files")
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            for i, row in df.iterrows():
                row_location = row.iloc[1]  # Use iloc to avoid deprecation warning
                
                for col in df.columns[2:]:  # Skip first two columns
                    value = row[col]
                    if not pd.isna(value) and value != '':
                        try:
                            fare = float(value)
                            final_tuples.append((row_location, col, fare))
                        except (ValueError, TypeError):
                            continue
                            
        except Exception as e:
            print(f"⚠️ Error processing file {file_path}: {e}")
            continue
    
    print(f"📊 Loaded {len(final_tuples)} transport routes")
    
    # Extract unique places
    unique_places = set()
    for from_place, to_place, fare in final_tuples:
        unique_places.add(from_place)
        unique_places.add(to_place)
    
    print(f"📍 Found {len(unique_places)} unique places")
    
    # Place name normalization (from ingest.py)
    print("🔄 Normalizing place names...")
    
    places = list(unique_places)
    groups = []
    visited = set()
    
    # Group similar place names
    for i, place in enumerate(places):
        if place in visited:
            continue
        # Find close matches with a cutoff for similarity
        matches = difflib.get_close_matches(place, places, n=10, cutoff=0.88)
        group = set(matches)
        visited.update(group)
        if len(group) > 1:
            groups.append(group)
    
    print(f"🔗 Found {len(groups)} groups of similar place names")
    
    # Create a mapping from each alias to the most suitable (canonical) name in each group
    def choose_canonical(group):
        """Choose the canonical name for a group of similar places"""
        # Prefer capitalized, longest, or most common spelling
        # Here, choose the most common spelling (by occurrence in edges), fallback to sorted
        counts = {}
        for a, b, _ in final_tuples:
            for name in group:
                if a == name or b == name:
                    counts[name] = counts.get(name, 0) + 1
        if counts:
            # Choose the name with the highest count, break ties by length, then alphabetically
            canonical = sorted(counts.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))[0][0]
        else:
            canonical = sorted(group, key=lambda x: (-len(x), x))[0]
        return canonical
    
    rename_map = {}
    for group in groups:
        canonical = choose_canonical(group)
        for name in group:
            rename_map[name] = canonical
    
    # Add manual mappings (from ingest.py)
    rename_map['Galkissa'] = 'Mount Lavinia'
    
    print(f"🔄 Created {len(rename_map)} place name mappings")
    
    # Remove .1 suffix from place names (from ingest.py)
    def remove_dot1(name):
        """Remove .1 suffix and find matching place name"""
        if name.endswith('.1'):
            base = name[:-2]
            # Find the matching place without '.1'
            for p in places:
                if p.lower() == base.lower():
                    return p
            return base
        return name
    
    # Apply all transformations to final_tuples
    print("🔄 Applying place name transformations...")
    transformed_tuples = []
    for from_place, to_place, fare in final_tuples:
        # Apply .1 removal
        clean_from = remove_dot1(from_place)
        clean_to = remove_dot1(to_place)
        
        # Apply normalization mapping
        final_from = rename_map.get(clean_from, clean_from)
        final_to = rename_map.get(clean_to, clean_to)
        
        transformed_tuples.append((final_from, final_to, fare))
    
    print(f"✅ Transformed {len(transformed_tuples)} routes with normalized place names")
    
    # Connection details
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "20665130@mM"))
    # driver = GraphDatabase.driver("bolt://52.54.129.206:7687", auth=("neo4j", "legends-bytes-linen"))
    
    # Clear existing data first
    print("🗑️ Clearing existing data...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    
    # Clean and validate data
    cleaned_tuples = []
    for i, (from_place, to_place, fare) in enumerate(transformed_tuples):
        # Skip invalid entries
        if pd.isna(from_place) or pd.isna(to_place) or pd.isna(fare):
            continue
        if from_place == '' or to_place == '':
            continue
        if from_place == to_place:  # Skip self-loops
            continue
            
        try:
            # Ensure fare is numeric
            fare_value = float(fare)
            if fare_value <= 0:  # Skip invalid fares
                continue
                
            cleaned_tuples.append((str(from_place).strip(), str(to_place).strip(), fare_value))
        except (ValueError, TypeError):
            continue
    
    print(f"✅ Cleaned data: {len(cleaned_tuples)} valid edges")
    
    # Create unique relationships with route_id to preserve duplicates
    print("🔄 Creating unique relationships with route IDs...")
    
    def insert_routes_with_ids(tx, routes_batch):
        """Insert routes with unique IDs to preserve duplicates"""
        route_data = []
        for i, (from_place, to_place, fare) in enumerate(routes_batch):
            route_data.append({
                'route_id': f"route_{i}",
                'from_place': from_place,
                'to_place': to_place,
                'fare': fare
            })
        
        # Create unique relationships with route_id property
        tx.run("""
            UNWIND $routes AS route
            MERGE (a:Place {name: route.from_place})
            MERGE (b:Place {name: route.to_place})
            CREATE (a)-[r:Fare {route_id: route.route_id, fare: route.fare}]->(b)
        """, routes=route_data)
    
    # Process all data in batches
    batch_size = 500
    total_edges = len(cleaned_tuples)
    start_time = time.time()
    processed = 0
    
    with driver.session() as session:
        for i in range(0, total_edges, batch_size):
            batch = cleaned_tuples[i:i + batch_size]
            
            try:
                session.execute_write(insert_routes_with_ids, batch)
                processed += len(batch)
                
                # Show progress
                progress = (processed / total_edges) * 100
                print(f"📊 Progress: {processed}/{total_edges} ({progress:.1f}%)")
                
                # Small delay
                time.sleep(0.05)
                
            except Exception as e:
                print(f"❌ Error processing batch {i//batch_size + 1}: {e}")
                continue
    
    end_time = time.time()
    
    # Final verification
    print("\n🔍 Final verification...")
    with driver.session() as session:
        # Count total routes
        result = session.run("MATCH ()-[r:Fare]->() RETURN count(r) as route_count")
        route_count = result.single()["route_count"]
        
        # Count total places
        result = session.run("MATCH (p:Place) RETURN count(p) as place_count")
        place_count = result.single()["place_count"]
        
        # Get fare statistics
        result = session.run("""
            MATCH ()-[r:Fare]->() 
            RETURN 
                min(r.fare) as min_fare,
                max(r.fare) as max_fare,
                avg(r.fare) as avg_fare
        """)
        stats = result.single()
        
        # Count unique route pairs (same from/to but different fares)
        result = session.run("""
            MATCH (a)-[r:Fare]->(b)
            WITH a, b, collect(r.fare) as fares
            RETURN count(*) as unique_route_pairs
        """)
        unique_pairs = result.single()["unique_route_pairs"]
        
        # Count routes with multiple fare options
        result = session.run("""
            MATCH (a)-[r:Fare]->(b)
            WITH a, b, collect(r.fare) as fares
            WHERE size(fares) > 1
            RETURN count(*) as routes_with_multiple_fares
        """)
        multiple_fares = result.single()["routes_with_multiple_fares"]
        
        # Show some examples of normalized place names
        result = session.run("""
            MATCH (p:Place)
            WHERE p.name CONTAINS 'Colombo' OR p.name CONTAINS 'Mount' OR p.name CONTAINS 'Galkissa'
            RETURN p.name as place_name
            LIMIT 10
        """)
        normalized_places = [record['place_name'] for record in result]
    
    print(f"✅ COMPLETE ingestion finished in {end_time - start_time:.2f} seconds")
    print(f"📊 Final Results:")
    print(f"   • Total Places: {place_count}")
    print(f"   • Total Routes: {route_count}")
    print(f"   • Unique Route Pairs: {unique_pairs}")
    print(f"   • Routes with Multiple Fares: {multiple_fares}")
    print(f"   • Minimum Fare: Rs. {stats['min_fare']:.2f}")
    print(f"   • Maximum Fare: Rs. {stats['max_fare']:.2f}")
    print(f"   • Average Fare: Rs. {stats['avg_fare']:.2f}")
    
    print(f"\n📍 Sample Normalized Place Names:")
    for place in normalized_places:
        print(f"   • {place}")
    
    # Show some sample routes with multiple fares
    print(f"\n📋 Sample Routes with Multiple Fare Options:")
    with driver.session() as session:
        result = session.run("""
            MATCH (a)-[r:Fare]->(b)
            WITH a, b, collect(r.fare) as fares
            WHERE size(fares) > 1
            RETURN a.name as from_place, b.name as to_place, fares
            LIMIT 5
        """)
        for record in result:
            fares_str = ", ".join([f"Rs. {f}" for f in record['fares']])
            print(f"   • {record['from_place']} -> {record['to_place']}: {fares_str}")
    
    # Show some sample individual routes
    print(f"\n📋 Sample Individual Routes:")
    with driver.session() as session:
        result = session.run("""
            MATCH (a)-[r:Fare]->(b)
            RETURN a.name as from_place, b.name as to_place, r.fare as fare, r.route_id as route_id
            LIMIT 10
        """)
        for record in result:
            print(f"   • {record['from_place']} -> {record['to_place']}: Rs. {record['fare']} (ID: {record['route_id']})")
    
    driver.close()
    return route_count

def main():
    """Main function"""
    print("🚌 Aligned Neo4j Transport Data Ingestion (All Routes + Normalization)")
    print("=" * 70)
    
    try:
        # Test connection first
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "20665130@mM"))
        # driver = GraphDatabase.driver("bolt://52.54.129.206:7687", auth=("neo4j", "legends-bytes-linen"))
        with driver.session() as session:
            session.run("RETURN 1")
        print("✅ Connected to Neo4j database")
        driver.close()
        
        # Run complete ingestion with normalization
        route_count = ingest_all_routes_with_normalization()
        
        if route_count:
            print(f"\n🎉 SUCCESS! Ingested {route_count} transport routes with place normalization!")
            print("💡 All routes are now preserved with unique route IDs and normalized place names")
        else:
            print("\n❌ Ingestion failed or no data was processed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure Neo4j is running and credentials are correct")

if __name__ == "__main__":
    main()
