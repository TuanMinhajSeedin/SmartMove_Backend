#!/usr/bin/env python3
"""
Quick start script for Schedule Ingestion App
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from schedule_ingestion_app import app

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Schedule Data Ingestion App")
    print("=" * 60)
    print(f"📊 Neo4j URI: bolt://localhost:7687")
    print(f"👤 Neo4j User: neo4j")
    print(f"🌐 Server: http://localhost:5001")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)


