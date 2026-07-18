#!/usr/bin/env python3
"""
Optimized Neo4j Data Ingestion Script
Handles large datasets efficiently with batch processing and progress tracking
"""

import os
import glob
import pandas as pd
from neo4j import GraphDatabase
from typing import List, Tuple
import time
from tqdm import tqdm

class OptimizedNeo4jIngestion:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.batch_size = 1000  # Process in batches of 1000
        
    def close(self):
        """Close the database connection"""
        self.driver.close()
    
    def load_transport_data(self, folder_path: str) -> List[Tuple[str, str, float]]:
        """
        Load all transport data from CSV files
        
        Args:
            folder_path: Path to the folder containing CSV files
            
        Returns:
            List of tuples (from_place, to_place, fare)
        """
        print(f"📁 Loading data from: {folder_path}")
        
        final_tuples = []
        csv_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
        
        print(f"📊 Found {len(csv_files)} CSV files")
        
        for file_path in tqdm(csv_files, desc="Processing CSV files"):
            try:
                df = pd.read_csv(file_path)
                
                for i, row in df.iterrows():
                    row_location = row[1]  # Assuming second column is the location
                    
                    for col in df.columns[2:]:  # Skip first two columns
                        value = row[col]
                        if not pd.isna(value) and value != '':
                            try:
                                # Convert to float to ensure numeric value
                                fare = float(value)
                                final_tuples.append((row_location, col, fare))
                            except (ValueError, TypeError):
                                # Skip non-numeric values
                                continue
                                
            except Exception as e:
                print(f"⚠️ Error processing file {file_path}: {e}")
                continue
        
        print(f"✅ Loaded {len(final_tuples)} transport routes")
        return final_tuples
    
    def standardize_location_names(self, tuples: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        """
        Standardize location names by removing duplicates and cleaning data
        
        Args:
            tuples: List of (from_place, to_place, fare) tuples
            
        Returns:
            Standardized list of tuples
        """
        print("🔧 Standardizing location names...")
        
        # Get all unique places
        unique_places = set()
        for from_place, to_place, fare in tuples:
            unique_places.add(from_place)
            unique_places.add(to_place)
        
        places = list(unique_places)
        print(f"📍 Found {len(places)} unique places")
        
        # Group similar places together
        groups = []
        visited = set()
        
        for i, place in enumerate(places):
            if place in visited:
                continue
                
            # Find close matches with similarity threshold
            matches = [place]
            for j, other_place in enumerate(places[i+1:], i+1):
                if other_place in visited:
                    continue
                    
                # Simple similarity check (you can use fuzzywuzzy for better matching)
                if self._are_similar_places(place, other_place):
                    matches.append(other_place)
                    visited.add(other_place)
            
            visited.add(place)
            groups.append(matches)
        
        # Create rename mapping
        rename_map = {}
        for group in groups:
            if len(group) > 1:
                # Use the first place as the canonical name
                canonical = group[0]
                for place in group[1:]:
                    rename_map[place] = canonical
        
        print(f"🔄 Created {len(rename_map)} location mappings")
        
        # Apply renaming
        standardized_tuples = []
        for from_place, to_place, fare in tuples:
            new_from = rename_map.get(from_place, from_place)
            new_to = rename_map.get(to_place, to_place)
            
            # Remove .1 suffixes
            new_from = self._remove_dot_suffix(new_from)
            new_to = self._remove_dot_suffix(new_to)
            
            standardized_tuples.append((new_from, new_to, fare))
        
        print(f"📊 After standardization: {len(standardized_tuples)} routes")
        return standardized_tuples
    
    def _are_similar_places(self, place1: str, place2: str) -> bool:
        """Check if two place names are similar (simple implementation)"""
        # Remove common suffixes and convert to lowercase
        p1 = place1.lower().replace(' ', '').replace('-', '')
        p2 = place2.lower().replace(' ', '').replace('-', '')
        
        # Check if one is contained in the other or if they're very similar
        if p1 in p2 or p2 in p1:
            return True
        
        # Check for common misspellings
        if len(p1) > 3 and len(p2) > 3:
            # Simple edit distance check
            if abs(len(p1) - len(p2)) <= 2:
                matches = sum(c1 == c2 for c1, c2 in zip(p1, p2))
                similarity = matches / max(len(p1), len(p2))
                return similarity > 0.8
        
        return False
    
    def _remove_dot_suffix(self, name: str) -> str:
        """Remove .1 suffix from place names"""
        if name.endswith('.1'):
            return name[:-2]
        return name
    
    def clear_database(self):
        """Clear all existing transport data"""
        print("🗑️ Clearing existing transport data...")
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        print("✅ Database cleared")
    
    def bulk_insert_routes(self, routes: List[Tuple[str, str, float]]):
        """
        Insert routes in batches for better performance
        
        Args:
            routes: List of (from_place, to_place, fare) tuples
        """
        print(f"🚀 Starting bulk insertion of {len(routes)} routes...")
        
        # Process in batches
        total_batches = (len(routes) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=len(routes), desc="Inserting routes") as pbar:
            for i in range(0, len(routes), self.batch_size):
                batch = routes[i:i + self.batch_size]
                
                try:
                    self._insert_batch(batch)
                    pbar.update(len(batch))
                    
                    # Small delay to prevent overwhelming the database
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"❌ Error inserting batch {i//self.batch_size + 1}: {e}")
                    # Continue with next batch
                    pbar.update(len(batch))
                    continue
        
        print("✅ Bulk insertion completed!")
    
    def _insert_batch(self, routes: List[Tuple[str, str, float]]):
        """Insert a batch of routes using UNWIND for efficiency"""
        
        with self.driver.session() as session:
            # Prepare data for UNWIND
            route_data = []
            for from_place, to_place, fare in routes:
                route_data.append({
                    'from_place': from_place,
                    'to_place': to_place,
                    'fare': fare
                })
            
            # Use UNWIND for efficient batch insertion
            # Create relationships with unique IDs to avoid overwriting duplicates
            session.run("""
                UNWIND $routes AS route
                MERGE (a:Place {name: route.from_place})
                MERGE (b:Place {name: route.to_place})
                CREATE (a)-[r:Fare {fare: route.fare}]->(b)
            """, routes=route_data)
    
    def verify_ingestion(self):
        """Verify the data was ingested correctly"""
        print("🔍 Verifying ingestion...")
        
        with self.driver.session() as session:
            # Count places
            places_result = session.run("MATCH (p:Place) RETURN count(p) as place_count")
            place_count = places_result.single()["place_count"]
            
            # Count routes
            routes_result = session.run("MATCH ()-[r:Fare]->() RETURN count(r) as route_count")
            route_count = routes_result.single()["route_count"]
            
            # Get fare statistics
            stats_result = session.run("""
                MATCH ()-[r:Fare]->() 
                RETURN 
                    min(r.fare) as min_fare,
                    max(r.fare) as max_fare,
                    avg(r.fare) as avg_fare,
                    count(r) as total_routes
            """)
            stats = stats_result.single()
            
            print("📊 Ingestion Verification Results:")
            print(f"   • Total Places: {place_count}")
            print(f"   • Total Routes: {route_count}")
            print(f"   • Minimum Fare: Rs. {stats['min_fare']:.2f}")
            print(f"   • Maximum Fare: Rs. {stats['max_fare']:.2f}")
            print(f"   • Average Fare: Rs. {stats['avg_fare']:.2f}")
            
            return {
                'places': place_count,
                'routes': route_count,
                'min_fare': stats['min_fare'],
                'max_fare': stats['max_fare'],
                'avg_fare': stats['avg_fare']
            }

def main():
    """Main function to run the optimized ingestion"""
    
    # Configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "20665130@mM"
    
    # Alternative: Use remote Neo4j
    # NEO4J_URI = "bolt://52.54.129.206:7687"
    # NEO4J_PASSWORD = "legends-bytes-linen"
    
    DATA_FOLDER = "artifacts/public_transport_data/translated_data/Fares"
    
    print("🚌 Optimized Neo4j Transport Data Ingestion")
    print("=" * 50)
    
    try:
        # Initialize ingestion
        ingestion = OptimizedNeo4jIngestion(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Test connection
        with ingestion.driver.session() as session:
            session.run("RETURN 1")
        print("✅ Connected to Neo4j database")
        
        # Load data
        routes = ingestion.load_transport_data(DATA_FOLDER)
        
        if not routes:
            print("❌ No data found to ingest")
            return
        
        # Standardize location names
        routes = ingestion.standardize_location_names(routes)
        
        # Clear existing data (optional - comment out if you want to keep existing data)
        clear_db = input("🗑️ Clear existing database? (y/N): ").lower().strip()
        if clear_db == 'y':
            ingestion.clear_database()
        
        # Insert data
        start_time = time.time()
        ingestion.bulk_insert_routes(routes)
        end_time = time.time()
        
        print(f"⏱️ Total ingestion time: {end_time - start_time:.2f} seconds")
        
        # Verify ingestion
        stats = ingestion.verify_ingestion()
        
        print("\n🎉 Data ingestion completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        
    finally:
        if 'ingestion' in locals():
            ingestion.close()

if __name__ == "__main__":
    main()
