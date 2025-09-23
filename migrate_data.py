#!/usr/bin/env python3

"""
Simple migration script to move data from local SQLite to Supabase.
"""

import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime
from storage import (
    get_db_path, init_database,
    insert_weight_for_user, insert_lbm_for_user, 
    set_height_for_user, set_preferences_for_user,
    _USE_SQLALCHEMY
)

def read_local_data():
    """Read data from local SQLite database."""
    db_path = get_db_path()
    
    if not os.path.exists(db_path):
        print(f"❌ Local database not found at {db_path}")
        return None
    
    print(f"📖 Reading data from {db_path}")
    
    conn = sqlite3.connect(db_path)
    
    # Read all tables
    tables = {}
    table_names = ['weights', 'lbm', 'user_settings', 'user_preferences']
    
    for table_name in table_names:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            tables[table_name] = df
            print(f"   📊 {table_name}: {len(df)} rows")
        except Exception as e:
            print(f"   ⚠️  {table_name}: {e}")
            tables[table_name] = pd.DataFrame()
    
    conn.close()
    return tables

def migrate_user_data(data):
    """Migrate user data to Supabase."""
    print("\n🚀 Migrating user data...")
    
    # Get all unique user IDs
    all_user_ids = set()
    for table_name in ['weights', 'lbm', 'user_settings', 'user_preferences']:
        df = data.get(table_name, pd.DataFrame())
        if not df.empty and 'user_id' in df.columns:
            all_user_ids.update(df['user_id'].unique())
    
    if not all_user_ids:
        print("   ℹ️  No user data found to migrate")
        return
    
    print(f"   👥 Found {len(all_user_ids)} users to migrate")
    
    for user_id in all_user_ids:
        print(f"\n   📝 Migrating user: {user_id}")
        
        # Migrate weights
        weights_df = data.get('weights', pd.DataFrame())
        if not weights_df.empty:
            user_weights = weights_df[weights_df['user_id'] == user_id]
            if not user_weights.empty:
                print(f"      ⚖️  Migrating {len(user_weights)} weight entries")
                for _, row in user_weights.iterrows():
                    try:
                        date = datetime.fromisoformat(row['date'])
                        weight = float(row['weight'])
                        insert_weight_for_user(user_id, date, weight)
                    except Exception as e:
                        print(f"         ❌ Error: {e}")
        
        # Migrate LBM data
        lbm_df = data.get('lbm', pd.DataFrame())
        if not lbm_df.empty:
            user_lbm = lbm_df[lbm_df['user_id'] == user_id]
            if not user_lbm.empty:
                print(f"      🏋️  Migrating {len(user_lbm)} LBM entries")
                for _, row in user_lbm.iterrows():
                    try:
                        date = datetime.fromisoformat(row['date'])
                        lbm_value = float(row['lbm'])
                        insert_lbm_for_user(user_id, date, lbm_value)
                    except Exception as e:
                        print(f"         ❌ Error: {e}")
        
        # Migrate user settings
        settings_df = data.get('user_settings', pd.DataFrame())
        if not settings_df.empty:
            user_settings = settings_df[settings_df['user_id'] == user_id]
            if not user_settings.empty:
                height = float(user_settings.iloc[0]['height'])
                print(f"      📏 Migrating height: {height} inches")
                try:
                    set_height_for_user(user_id, height)
                except Exception as e:
                    print(f"         ❌ Error: {e}")
        
        # Migrate user preferences
        prefs_df = data.get('user_preferences', pd.DataFrame())
        if not prefs_df.empty:
            user_prefs = prefs_df[prefs_df['user_id'] == user_id]
            if not user_prefs.empty:
                prefs_row = user_prefs.iloc[0]
                print(f"      ⚙️  Migrating preferences")
                try:
                    set_preferences_for_user(
                        user_id,
                        prefs_row.get('confidence_interval', '1σ'),
                        bool(prefs_row.get('enable_forecast', 1)),
                        int(prefs_row.get('forecast_days', 30)),
                        int(prefs_row.get('residuals_bins', 15))
                    )
                except Exception as e:
                    print(f"         ❌ Error: {e}")

def main():
    """Main migration function."""
    print("=== Data Migration to Supabase ===")
    print()
    
    # Check if using Supabase
    if not _USE_SQLALCHEMY:
        print("❌ Not connected to Supabase!")
        print("Please run setup_supabase.py first")
        sys.exit(1)
    
    # Initialize database
    print("🔧 Initializing database...")
    init_database()
    
    # Read local data
    data = read_local_data()
    if data is None:
        sys.exit(1)
    
    # Migrate data
    migrate_user_data(data)
    
    print("\n✅ Migration complete!")
    print("\nYour data is now in Supabase!")
    print("You can run your Streamlit app and it will use the cloud database.")
    print("\nNote: Your local SQLite file is still there as a backup.")

if __name__ == "__main__":
    main()
