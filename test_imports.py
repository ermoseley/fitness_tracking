#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

try:
    import streamlit as st
    print("✓ streamlit imported successfully")
except ImportError as e:
    print(f"✗ streamlit import failed: {e}")

try:
    import pandas as pd
    print("✓ pandas imported successfully")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")

try:
    import numpy as np
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    import plotly.graph_objects as go
    print("✓ plotly.graph_objects imported successfully")
except ImportError as e:
    print(f"✗ plotly.graph_objects import failed: {e}")

try:
    import plotly.express as px
    print("✓ plotly.express imported successfully")
except ImportError as e:
    print(f"✗ plotly.express import failed: {e}")

try:
    from plotly.subplots import make_subplots
    print("✓ plotly.subplots imported successfully")
except ImportError as e:
    print(f"✗ plotly.subplots import failed: {e}")

try:
    import scipy
    print("✓ scipy imported successfully")
except ImportError as e:
    print(f"✗ scipy import failed: {e}")

try:
    import matplotlib
    print("✓ matplotlib imported successfully")
except ImportError as e:
    print(f"✗ matplotlib import failed: {e}")

try:
    import sqlite3
    print("✓ sqlite3 imported successfully")
except ImportError as e:
    print(f"✗ sqlite3 import failed: {e}")

try:
    from storage import init_database
    print("✓ storage module imported successfully")
except ImportError as e:
    print(f"✗ storage module import failed: {e}")

try:
    from weight_tracker import WeightEntry
    print("✓ weight_tracker module imported successfully")
except ImportError as e:
    print(f"✗ weight_tracker module import failed: {e}")

try:
    from kalman import WeightKalmanFilter
    print("✓ kalman module imported successfully")
except ImportError as e:
    print(f"✗ kalman module import failed: {e}")

print("\nAll import tests completed.")
