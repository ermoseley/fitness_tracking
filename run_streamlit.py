#!/usr/bin/env python3
"""
Launcher script for the Streamlit Weight Tracker
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit Weight Tracker application"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "streamlit_app.py")
    
    # Check if the app file exists
    if not os.path.exists(app_path):
        print(f"Error: streamlit_app.py not found at {app_path}")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Error: Streamlit is not installed. Please run:")
        print("pip install -r requirements_streamlit.txt")
        sys.exit(1)
    
    # Launch the Streamlit app
    print("Starting Weight Tracker...")
    print("The app will open in your default web browser.")
    print("Press Ctrl+C to stop the application.")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ], cwd=script_dir)
    except KeyboardInterrupt:
        print("\nWeight Tracker stopped.")
    except Exception as e:
        print(f"Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
