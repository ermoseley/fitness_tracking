#!/usr/bin/env python3
"""
Database migration script to add the default_plot_range_days column
to existing user_preferences tables.
"""

import sqlite3
import os
from storage import get_db_path, _USE_SQLALCHEMY, get_engine

def migrate_sqlite_database():
    """Migrate SQLite database to add the new column"""
    db_path = get_db_path()
    print(f"Migrating SQLite database at: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if the column already exists
        cursor.execute("PRAGMA table_info(user_preferences)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'default_plot_range_days' not in columns:
            print("Adding default_plot_range_days column to user_preferences table...")
            cursor.execute("ALTER TABLE user_preferences ADD COLUMN default_plot_range_days INTEGER NOT NULL DEFAULT 60")
            conn.commit()
            print("✅ Migration completed successfully!")
        else:
            print("✅ Column already exists, no migration needed.")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

def migrate_postgres_database():
    """Migrate PostgreSQL database to add the new column"""
    if not _USE_SQLALCHEMY:
        print("Not using SQLAlchemy, skipping PostgreSQL migration")
        return
        
    try:
        from sqlalchemy import text
        engine = get_engine()
        
        with engine.begin() as conn:
            # Check if the column already exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'user_preferences' 
                AND column_name = 'default_plot_range_days'
            """))
            
            if result.fetchone() is None:
                print("Adding default_plot_range_days column to user_preferences table...")
                conn.execute(text("""
                    ALTER TABLE user_preferences 
                    ADD COLUMN default_plot_range_days INTEGER NOT NULL DEFAULT 60
                """))
                print("✅ PostgreSQL migration completed successfully!")
            else:
                print("✅ Column already exists, no migration needed.")
                
    except Exception as e:
        print(f"❌ PostgreSQL migration failed: {e}")

def main():
    """Run the appropriate migration based on database type"""
    print("🔧 Starting database migration...")
    
    if _USE_SQLALCHEMY:
        print("Using PostgreSQL/SQLAlchemy database")
        migrate_postgres_database()
    else:
        print("Using SQLite database")
        migrate_sqlite_database()
    
    print("🎉 Migration process completed!")

if __name__ == "__main__":
    main()
