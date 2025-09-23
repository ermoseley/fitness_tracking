#!/usr/bin/env python3

"""
Setup script to configure Supabase connection for the fitness tracker.
"""

import os
import sys
from storage import _USE_SQLALCHEMY, _engine

def test_connection():
    """Test the connection to Supabase."""
    print("Testing Supabase connection...")
    
    if not _USE_SQLALCHEMY or not _engine:
        print("❌ Error: DATABASE_URL not configured or SQLAlchemy not available")
        print("\nPlease set your DATABASE_URL environment variable:")
        print("export DATABASE_URL='postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres'")
        return False
    
    try:
        from sqlalchemy import text
        with _engine.begin() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            print("✅ Connection successful!")
            return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nPlease check:")
        print("1. Your DATABASE_URL is correct")
        print("2. Your Supabase project is running")
        print("3. Your password is correct")
        return False

def setup_database():
    """Initialize the database tables."""
    print("\nSetting up database tables...")
    
    try:
        from storage import init_database
        init_database()
        print("✅ Database tables created successfully!")
        return True
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def main():
    """Main setup function."""
    print("=== Supabase Setup for Fitness Tracker ===")
    print()
    
    # Check if DATABASE_URL is set
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL environment variable not set")
        print("\nPlease run:")
        print("export DATABASE_URL='postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres'")
        print("\nReplace [PASSWORD] with your actual database password")
        print("Replace [PROJECT] with your project reference")
        sys.exit(1)
    
    print(f"✅ DATABASE_URL is configured")
    print(f"   Using: {database_url[:50]}...")
    
    # Test connection
    if not test_connection():
        sys.exit(1)
    
    # Setup database
    if not setup_database():
        sys.exit(1)
    
    print("\n🎉 Setup complete!")
    print("\nYour fitness tracker is now connected to Supabase!")
    print("You can run your Streamlit app and it will use the cloud database.")

if __name__ == "__main__":
    main()
