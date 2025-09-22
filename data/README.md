# Data Directory

This directory contains user data and should not be tracked by git.

## Structure

- `fitness_tracker.db` - Main SQLite database
- `users/` - Per-user data directories
  - `{user_id}/` - Individual user data
    - `weights.csv` - Weight measurements
    - `lbm.csv` - Lean body mass measurements  
    - `height.txt` - User height in inches

## Sample Data

The `users/sample/` directory contains example data files for testing purposes.

## Important

- User data is private and should never be committed to version control
- The database is automatically created when the app runs
- Each user gets their own isolated data directory
