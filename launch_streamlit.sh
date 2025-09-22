#!/bin/bash
# Launch script for Streamlit Weight Tracker

echo "Starting Weight Tracker..."
echo "The app will open in your default web browser."
echo "Press Ctrl+C to stop the application."
echo ""

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "Error: Streamlit is not installed. Please run:"
    echo "pip install -r requirements_streamlit.txt"
    exit 1
fi

# Launch the Streamlit app
python3 -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost
