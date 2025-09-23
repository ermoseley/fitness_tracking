#!/usr/bin/env python3

"""
Simple connection test for Supabase.
"""

import os
import psycopg2

def test_connection():
    """Test direct connection to Supabase."""
    database_url = os.environ.get("DATABASE_URL")
    
    if not database_url:
        print("❌ DATABASE_URL not set")
        return False
    
    print(f"Testing connection to: {database_url[:50]}...")
    
    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print("✅ Connection successful!")
        print(f"Test query result: {result}")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
