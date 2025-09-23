#!/usr/bin/env python3
"""
Streamlit Fitness Tracking Application
Port of the BodyMetrics GUI to Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, date, timedelta
from typing import List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import re
import uuid

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our existing modules
from weight_tracker import (
    WeightEntry, parse_datetime, parse_date,
    compute_time_aware_ema, fit_time_weighted_linear_regression,
    aggregate_entries
)
from storage import (
    init_database, get_weights_for_user, insert_weight_for_user, replace_weights_for_user,
    get_lbm_df_for_user, insert_lbm_for_user, replace_lbm_for_user,
    get_height_for_user, set_height_for_user,
    get_preferences_for_user, set_preferences_for_user,
    delete_weight_entry, delete_lbm_entry
)
from auth import init_auth_tables, require_auth, get_current_user, logout
from kalman import (
    WeightKalmanFilter, KalmanState, run_kalman_filter, 
    run_kalman_smoother, create_kalman_plot, create_bodyfat_plot_from_kalman,
    create_bmi_plot_from_kalman, create_ffmi_plot_from_kalman
)


def get_mobile_friendly_legend_config():
    """
    Returns a mobile-friendly legend configuration for Plotly charts.
    Positions legend horizontally below the chart for better mobile experience.
    """
    return dict(
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="center",
        x=0.5,
        font=dict(size=10)
    )


def get_mobile_friendly_layout_config():
    """
    Returns layout configuration with proper margins for mobile-friendly legends.
    """
    return dict(
        legend=get_mobile_friendly_legend_config(),
        margin=dict(b=80)  # Bottom margin for legend space
    )


def get_velocity_mobile_layout_config():
    """
    Returns layout configuration for velocity plots with slightly lower legend positioning.
    """
    return dict(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,  # Lower than default for velocity plots
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        margin=dict(b=90)  # Extra bottom margin for velocity plots
    )


def get_residuals_mobile_layout_config():
    """
    Returns layout configuration for residuals plots with extra space for legend.
    """
    return dict(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        margin=dict(b=100)  # Extra bottom margin for residuals plots with legend
    )


def is_mobile_device():
    """
    Simple mobile detection based on Streamlit's user agent.
    Returns True if the device appears to be mobile.
    """
    try:
        # Streamlit doesn't directly expose user agent, but we can use CSS media queries
        # For now, we'll assume mobile for all devices and optimize accordingly
        return True
    except:
        return True


def get_default_plot_range(entries):
    """
    Calculate the default plot range based on user preference and available data.
    Returns (start_date, end_date) or (None, None) for full range.
    """
    if not entries or len(entries) == 0:
        return None, None
    
    # Get the user's preferred default range
    default_range_days = float(st.session_state.get('default_plot_range_days', 60))
    
    # Calculate the total data range
    earliest_date = entries[0].entry_datetime
    latest_date = entries[-1].entry_datetime
    span_days = (latest_date - earliest_date).total_seconds() / 86400.0
    
    # If we have less than the default range of data, show all data
    if span_days < default_range_days:
        return None, None
    
    # Otherwise, show the last N days
    start_date = latest_date - timedelta(days=default_range_days)
    return start_date, latest_date


def get_weight_yaxis_range(entries, start_date=None, end_date=None, padding_factor=None):
    """
    Calculate an appropriate Y-axis range for weight charts.
    Returns (y_min, y_max) or None if no range should be set.
    
    Args:
        entries: List of weight entries
        start_date: Optional start date to filter entries
        end_date: Optional end date to filter entries  
        padding_factor: Factor for padding above/below data (default 0.1 = 10%)
    """
    if not entries:
        return None
    
    # Filter entries by date range if specified
    filtered_entries = entries
    if start_date or end_date:
        if start_date:
            filtered_entries = [e for e in filtered_entries if e.entry_datetime >= start_date]
        if end_date:
            filtered_entries = [e for e in filtered_entries if e.entry_datetime <= end_date]
    
    if not filtered_entries:
        return None
    
    # Get weight values
    weights = [e.weight for e in filtered_entries]
    if not weights:
        return None
    
    # Calculate range
    min_weight = min(weights)
    max_weight = max(weights)
    weight_range = max_weight - min_weight
    
    # Use user's padding preference or default
    if padding_factor is None:
        padding_factor = st.session_state.get('yaxis_padding_factor', 0.1)
    
    # Add padding
    padding = max(weight_range * padding_factor, 1.0)  # At least 1 lb padding
    y_min = min_weight - padding
    y_max = max_weight + padding
    
    return y_min, y_max

# Utilities
def get_confidence_multiplier(confidence_setting: str) -> float:
    """Get the confidence multiplier based on the setting"""
    if confidence_setting == "1σ":
        return 1.0
    elif confidence_setting == "95%":
        return 1.96
    else:
        return 1.0  # Default to 1σ

def ensure_py_datetime(dt: object) -> datetime:
    """Convert numpy/pandas datetime types to Python datetime."""
    try:
        import pandas as _pd  # local import to avoid hard dep in type context
    except Exception:
        _pd = None
    if isinstance(dt, np.datetime64):
        if _pd is not None:
            return _pd.to_datetime(dt).to_pydatetime()
        # Fallback: convert via ISO string
        return datetime.fromisoformat(str(dt))
    if _pd is not None and isinstance(dt, _pd.Timestamp):
        return dt.to_pydatetime()
    return dt  # assume already datetime

# ---------------------------
# Per-user data utilities
# ---------------------------

def sanitize_user_id(user_id: str) -> str:
    """Sanitize a user identifier to be filesystem-safe."""
    if not isinstance(user_id, str):
        user_id = str(user_id)
    # Allow letters, numbers, dash, underscore; replace others with dash
    cleaned = re.sub(r"[^A-Za-z0-9_\-]", "-", user_id.strip())
    # Collapse multiple dashes
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned or f"guest-{uuid.uuid4().hex[:8]}"

def get_user_data_dir() -> str:
    """Return the data directory path for the current user."""
    user_id = st.session_state.get("user_id")
    if not user_id:
        user_id = f"guest-{uuid.uuid4().hex[:8]}"
        st.session_state.user_id = user_id
    safe_user = sanitize_user_id(user_id)
    return os.path.join("data", "users", safe_user)

def ensure_user_files_exist() -> None:
    """Ensure the current user's data directory and files exist with correct headers."""
    data_dir = get_user_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    weights_path = os.path.join(data_dir, "weights.csv")
    lbm_path = os.path.join(data_dir, "lbm.csv")
    height_path = os.path.join(data_dir, "height.txt")

    # Create blank CSVs with headers if missing
    if not os.path.exists(weights_path):
        with open(weights_path, "w", newline="") as f:
            f.write("date,weight\n")
    if not os.path.exists(lbm_path):
        with open(lbm_path, "w", newline="") as f:
            f.write("date,lbm\n")
    # Initialize height if missing
    if not os.path.exists(height_path):
        try:
            default_height = float(st.session_state.get("height", 67.0))
        except Exception:
            default_height = 67.0
        with open(height_path, "w") as f:
            f.write(f"{default_height:.3f}")

# Page configuration
st.set_page_config(
    page_title="",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and mobile responsiveness
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Mobile-friendly improvements */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
            margin-bottom: 1rem;
        }
        
        /* Make plotly charts more mobile-friendly */
        .js-plotly-plot .plotly {
            overflow-x: auto;
        }
        
        /* Improve sidebar on mobile */
        .sidebar .sidebar-content {
            padding: 1rem 0.5rem;
        }
        
        /* Better spacing for metrics */
        .stMetric {
            margin-bottom: 0.5rem;
        }
    }
    
    /* Ensure plotly legends don't overflow on small screens */
    .plotly .legend {
        max-width: 100% !important;
        overflow-x: auto !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"guest-{uuid.uuid4().hex[:8]}"
if 'weights_data' not in st.session_state:
    st.session_state.weights_data = []
if 'lbm_data' not in st.session_state:
    st.session_state.lbm_data = []
if 'height' not in st.session_state:
    st.session_state.height = 67.0  # Default height in inches
if 'forecast_days' not in st.session_state:
    st.session_state.forecast_days = 30  # Default forecast duration
if 'enable_forecast' not in st.session_state:
    st.session_state.enable_forecast = True  # Default forecast enabled
if 'confidence_interval' not in st.session_state:
    st.session_state.confidence_interval = "1σ"  # Default to 1σ
if 'residuals_bins' not in st.session_state:
    st.session_state.residuals_bins = 15  # Default number of bins for residuals histogram
if 'default_plot_range_days' not in st.session_state:
    st.session_state.default_plot_range_days = 60  # Default plot range in days


def load_data_files():
    """Load data for current user from the SQLite database"""
    # Get the authenticated user ID
    user_id = get_current_user()
    if not user_id:
        st.error("No authenticated user found")
        return
    
    # Sanitize the user ID for database storage
    user_id = sanitize_user_id(user_id)
    st.session_state.user_id = user_id

    # One-time migration: if DB is empty but legacy per-user CSV files exist, import them
    try:
        has_weights = False
        try:
            existing = get_weights_for_user(user_id)
            has_weights = len(existing) > 0
        except Exception:
            has_weights = False

        if not has_weights:
            data_dir = os.path.join("data", "users", user_id)
            weights_csv = os.path.join(data_dir, "weights.csv")
            if os.path.exists(weights_csv):
                try:
                    df_w = pd.read_csv(weights_csv)
                    if 'date' in df_w.columns and 'weight' in df_w.columns and len(df_w) > 0:
                        replace_weights_for_user(user_id, df_w)
                except Exception:
                    # Fallback: try robust parser from weight_tracker
                    try:
                        from weight_tracker import load_entries as _load_entries
                        entries = _load_entries(weights_csv)
                        if entries:
                            df_w2 = pd.DataFrame({
                                'date': [e.entry_datetime for e in entries],
                                'weight': [e.weight for e in entries]
                            })
                            replace_weights_for_user(user_id, df_w2)
                    except Exception:
                        pass

            lbm_csv = os.path.join(data_dir, "lbm.csv")
            if os.path.exists(lbm_csv):
                try:
                    df_l = pd.read_csv(lbm_csv)
                    if 'date' in df_l.columns and 'lbm' in df_l.columns and len(df_l) > 0:
                        replace_lbm_for_user(user_id, df_l)
                except Exception:
                    pass

            height_txt = os.path.join(data_dir, "height.txt")
            if get_height_for_user(user_id) is None and os.path.exists(height_txt):
                try:
                    with open(height_txt, 'r') as f:
                        h_val = float(f.read().strip())
                    set_height_for_user(user_id, h_val)
                except Exception:
                    pass
    except Exception:
        # Migration is best-effort; ignore failures
        pass

    # Load weights from DB
    try:
        rows = get_weights_for_user(user_id)
        st.session_state.weights_data = [WeightEntry(dt, float(w)) for dt, w in rows]
        st.session_state.data_loaded = bool(st.session_state.weights_data)
    except Exception:
        st.session_state.weights_data = []
        st.session_state.data_loaded = False

    # Load LBM from DB
    try:
        st.session_state.lbm_data = get_lbm_df_for_user(user_id)
    except Exception:
        st.session_state.lbm_data = pd.DataFrame()

    # Load height from DB (default to 67.0 inches if not set)
    try:
        h = get_height_for_user(user_id)
        if h is None:
            # initialize with default height in DB
            default_height = float(st.session_state.get("height", 67.0))
            set_height_for_user(user_id, default_height)
            st.session_state.height = default_height
        else:
            st.session_state.height = float(h)
    except Exception:
        st.session_state.height = 67.0

    # Load user preferences and sync into session
    try:
        prefs = get_preferences_for_user(user_id)
        if prefs:
            st.session_state.confidence_interval = prefs.get("confidence_interval", st.session_state.confidence_interval)
            st.session_state.enable_forecast = bool(prefs.get("enable_forecast", st.session_state.enable_forecast))
            st.session_state.forecast_days = int(prefs.get("forecast_days", st.session_state.forecast_days))
            st.session_state.residuals_bins = int(prefs.get("residuals_bins", st.session_state.residuals_bins))
            st.session_state.default_plot_range_days = int(prefs.get("default_plot_range_days", st.session_state.default_plot_range_days))
        else:
            # Persist defaults for first-time users
            set_preferences_for_user(
                user_id,
                st.session_state.confidence_interval,
                bool(st.session_state.enable_forecast),
                int(st.session_state.forecast_days),
                int(st.session_state.residuals_bins),
                int(st.session_state.default_plot_range_days),
            )
    except Exception:
        pass

def save_height(height_inches: float):
    """Save height to database for current user"""
    user_id = get_current_user()
    if not user_id:
        st.error("No authenticated user found")
        return
    
    user_id = sanitize_user_id(user_id)
    set_height_for_user(user_id, float(height_inches))
    st.session_state.height = float(height_inches)

def main():
    # Initialize database and authentication on startup
    init_database()
    init_auth_tables()
    
    # Check authentication - if not authenticated, show login form and return
    if not require_auth():
        return
    
    # Load data for authenticated user
    load_data_files()
    
    # Add custom CSS to reduce spacing between metrics and captions
    st.markdown("""
    <style>
    .stCaption {
        margin-top: -10px !important;
        margin-bottom: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header"> BodyMetrics</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # User profile / logout
        st.header("👤 User Profile")
        current_user = get_current_user()
        if current_user:
            st.success(f"Logged in as: **{current_user}**")
            if st.button("🚪 Logout", type="secondary"):
                logout()
                return
        else:
            st.error("Not authenticated")
            return

        st.header("📊 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Dashboard", "➕ Add Entries", "📈 Weight Analysis", "📊 Composition Analysis", "⚙️ Settings", "📁 Data Management", "📖 User Guide", "⚠️ Disclaimer"]
        )
        
        st.header("📋 Quick Stats")
        if st.session_state.weights_data:
            latest_entry = st.session_state.weights_data[-1]
            st.metric("Latest Weight", f"{latest_entry.weight:.1f} lbs")
            st.metric("Total Entries", len(st.session_state.weights_data))
            st.metric("Date Range", f"{(latest_entry.entry_datetime - st.session_state.weights_data[0].entry_datetime).days} days")
        else:
            st.info("No weight data available")
    
    # Main content based on selected page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "➕ Add Entries":
        show_add_entries()
    elif page == "📈 Weight Analysis":
        show_weight_tracking()
    elif page == "📊 Composition Analysis":
        show_body_composition()
    elif page == "⚙️ Settings":
        show_settings()
    elif page == "📁 Data Management":
        show_data_management()
    elif page == "📖 User Guide":
        show_user_guide()
    elif page == "⚠️ Disclaimer":
        show_disclaimer()

def show_dashboard():
    """Main dashboard with overview metrics and charts"""
    st.header("📊 Dashboard Overview")
    
    if not st.session_state.weights_data:
        st.warning("No weight data available. Please add some weight entries first.")
        return
    
    # Calculate key metrics using Kalman filter (primary method)
    entries = st.session_state.weights_data
    latest_entry = entries[-1]
    first_entry = entries[0]
    
    # Run Kalman filter as primary analysis method
    try:
        kalman_states, kalman_dates = run_kalman_smoother(entries)
        
        if kalman_states:
            latest_kalman = kalman_states[-1]
            
            # Display key metrics from Kalman filter
            col1, col2, col3, col4 = st.columns(4)
            
            # Get confidence multiplier for error reporting
            ci_mult = get_confidence_multiplier(st.session_state.confidence_interval)
            
            with col1:
                st.metric(
                    "Current Weight Estimate",
                    f"{latest_kalman.weight:.1f} lbs"
                )
                st.caption(f"±{ci_mult * latest_kalman.weight_var**0.5:.1f}")
            
            with col2:
                velocity_per_week = 7 * latest_kalman.velocity
                st.metric(
                    "Weekly Rate",
                    f"{velocity_per_week:+.3f} lbs/week"
                )
                st.caption(f"±{ci_mult * 7 * (latest_kalman.velocity_var**0.5):.3f}")
            
            with col3:
                calorie_deficit = latest_kalman.velocity * 3500
                st.metric(
                    "Calorie Surplus/Deficit",
                    f"{calorie_deficit:+.0f} cal/day"
                )
                st.caption("Estimated")
            
            with col4:
                st.metric(
                    "Total Entries",
                    len(entries)
                )
                st.caption(f"{(latest_entry.entry_datetime - first_entry.entry_datetime).days} days")
            
            # Weight trend chart with Kalman filter and forecast extension
            st.subheader("Weight Trend")
            
            # Create dense, smooth curves using the same approach as the original
            from kalman import compute_kalman_mean_std_spline
            dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(kalman_states, kalman_dates)
            # Normalize datetime objects to pure Python datetimes
            dense_datetimes = [ensure_py_datetime(d) for d in dense_datetimes]
            
            # Add forecast extension if enabled
            if st.session_state.enable_forecast:
                # Calculate forecasts using the same approach as weight tracking page
                kf = WeightKalmanFilter(
                    initial_weight=latest_kalman.weight,
                    initial_velocity=latest_kalman.velocity,
                    initial_weight_var=latest_kalman.weight_var,
                    initial_velocity_var=latest_kalman.velocity_var,
                )
                kf.x = np.array([latest_kalman.weight, latest_kalman.velocity], dtype=float)
                kf.P = np.array([
                    [latest_kalman.weight_var, latest_kalman.weight_velocity_cov],
                    [latest_kalman.weight_velocity_cov, latest_kalman.velocity_var],
                ], dtype=float)
                
                # Generate forecast data using configurable duration
                forecast_days = int(st.session_state.forecast_days)
                latest_date = ensure_py_datetime(dense_datetimes[-1])
                forecast_dates = [latest_date + timedelta(days=int(i)) for i in range(1, forecast_days + 1)]
                forecast_weights = []
                forecast_uncertainties = []
                
                for i in range(1, forecast_days + 1):
                    weight, std = kf.forecast(float(i))
                    forecast_weights.append(weight)
                    forecast_uncertainties.append(std)
                
                # Combine historical and forecast data
                combined_dates = list(dense_datetimes) + list(forecast_dates)
                combined_means = list(dense_means) + list(forecast_weights)
                combined_stds = list(dense_stds) + list(forecast_uncertainties)
            else:
                # No forecast, just use historical data
                combined_dates = dense_datetimes
                combined_means = dense_means
                combined_stds = dense_stds
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add original data points
            dates = [entry.entry_datetime for entry in entries]
            weights = [entry.weight for entry in entries]
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=weights,
                mode='markers',
                name='Weight Measurements',
                marker=dict(size=8, color='blue', opacity=0.7)
            ))
            
            # Add dense smoothed estimate curve (historical portion)
            fig.add_trace(go.Scatter(
                x=dense_datetimes,
                y=dense_means,
                mode='lines',
                name='Smoothed Trend',
                line=dict(color='red', width=2)
            ))
            
            # Add forecast portion with dashed line and fading opacity
            if st.session_state.enable_forecast:
                # Create opacity gradient for forecast (fade from 1.0 to 0.3)
                forecast_opacity = np.linspace(1.0, 0.3, len(forecast_weights))
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_weights,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash'),
                    opacity=0.7  # Overall opacity for the trace
                ))
            
            # Add smooth confidence intervals (historical portion)
            ci_multiplier = get_confidence_multiplier(st.session_state.confidence_interval)
            hist_upper_bound = [mean + ci_multiplier * std for mean, std in zip(dense_means, dense_stds)]
            hist_lower_bound = [mean - ci_multiplier * std for mean, std in zip(dense_means, dense_stds)]
            
            fig.add_trace(go.Scatter(
                x=dense_datetimes,
                y=hist_upper_bound,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=dense_datetimes,
                y=hist_lower_bound,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name=f'{st.session_state.confidence_interval} Confidence',
                hoverinfo='skip'
            ))
            
            # Add forecast confidence intervals with fading opacity
            if st.session_state.enable_forecast:
                forecast_upper_bound = [mean + ci_multiplier * std for mean, std in zip(forecast_weights, forecast_uncertainties)]
                forecast_lower_bound = [mean - ci_multiplier * std for mean, std in zip(forecast_weights, forecast_uncertainties)]
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_upper_bound,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip',
                    opacity=0.5
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_lower_bound,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)',
                    name=f'Forecast {st.session_state.confidence_interval} CI',
                    hoverinfo='skip',
                    opacity=0.5
                ))
            
            # Note: We avoid add_vline with datetime here due to compatibility issues
            # in some Plotly versions. The forecast joins smoothly from the last
            # historical point, so the separation is visually clear without a marker.
            
            # Set default plot range based on user preference
            start_date, end_date = get_default_plot_range(entries)
            layout_config = get_mobile_friendly_layout_config()
            if start_date is not None and end_date is not None:
                layout_config['xaxis'] = {'range': [start_date, end_date]}
            
            # Set appropriate Y-axis range for weight data
            y_range = get_weight_yaxis_range(entries, start_date, end_date)
            if y_range:
                layout_config['yaxis'] = {'range': y_range}
            
            fig.update_layout(
                title="Weight Trend Analysis" + (f" + {forecast_days}-Day Forecast" if st.session_state.enable_forecast else ""),
                xaxis_title="Date",
                yaxis_title="Weight (lbs)",
                hovermode='x unified',
                height=500,
                **layout_config
            )
            if start_date is not None and end_date is not None:
                fig.update_xaxes(range=[start_date, end_date])
            if y_range:
                fig.update_yaxes(range=y_range)
            
            st.plotly_chart(fig, width='stretch')
            
            # Forecast summary metrics (only if enabled)
            if st.session_state.enable_forecast:
                st.subheader("🔮 Forecast Summary")
                
                # Calculate forecasts for summary metrics
                kf = WeightKalmanFilter(
                    initial_weight=latest_kalman.weight,
                    initial_velocity=latest_kalman.velocity,
                    initial_weight_var=latest_kalman.weight_var,
                    initial_velocity_var=latest_kalman.velocity_var,
                )
                kf.x = np.array([latest_kalman.weight, latest_kalman.velocity], dtype=float)
                kf.P = np.array([
                    [latest_kalman.weight_var, latest_kalman.weight_velocity_cov],
                    [latest_kalman.weight_velocity_cov, latest_kalman.velocity_var],
                ], dtype=float)
                
                # Forecast summary metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    week_forecast, week_std = kf.forecast(7.0)
                    st.metric(
                        "1-Week Forecast",
                        f"{week_forecast:.2f} lbs"
                    )
                    st.caption(f"±{ci_mult * week_std:.2f}")
                
                with col2:
                    forecast_days = st.session_state.forecast_days
                    forecast_value, forecast_std = kf.forecast(float(forecast_days))
                    st.metric(
                        f"{forecast_days}-Day Forecast",
                        f"{forecast_value:.2f} lbs"
                    )
                    st.caption(f"±{ci_mult * forecast_std:.2f}")
            
            # Recent entries table
            st.subheader("📋 Recent Entries")
            recent_entries = entries[-10:]  # Last 10 entries
            
            # Simple table without Kalman interpolation (for dashboard)
            df_recent = pd.DataFrame([
                {
                    'Date': entry.entry_datetime.strftime('%Y-%m-%d %H:%M'),
                    'Weight (lbs)': f"{entry.weight:.1f}",
                    'Days Since Start': f"{(entry.entry_datetime - first_entry.entry_datetime).days}"
                }
                for entry in recent_entries
            ])
            st.dataframe(df_recent, width='stretch')
            
        else:
            st.error("Failed to run trend analysis")
            return
            
    except Exception as e:
        st.error(f"Error running trend analysis: {e}")
        # Fallback to basic display
        st.metric("Latest Weight", f"{latest_entry.weight:.1f} lbs")
        st.metric("Total Entries", len(entries))
        return

def show_add_entries():
    """Dedicated page for adding new entries"""
    st.header("➕ Add New Entries")
    st.write("Add weight measurements, lean body mass (LBM), or body fat percentage data.")
    
    # Create tabs for different entry types
    tab1, tab2, tab3 = st.tabs(["Weight Entry", "LBM Entry", "Body Fat % Entry"])
    
    with tab1:
        st.write("**Add Weight Entry**")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Date and time input (default to today, 9:00 AM)
            entry_date = st.date_input("Date", value=date.today(), key="weight_date")
            entry_time = st.time_input("Time", value=datetime.strptime("09:00", "%H:%M").time(), key="weight_time")
            entry_datetime = datetime.combine(entry_date, entry_time)
        
        with col2:
            weight_value = st.number_input("Weight (lbs)", min_value=50.0, max_value=500.0, value=150.0, step=0.1, key="weight_value")
            
            if st.button("Add Weight Entry", type="primary", key="add_weight"):
                if weight_value > 0:
                    # Save to DB (per-user)
                    user_id = get_current_user()
                    if user_id:
                        insert_weight_for_user(sanitize_user_id(user_id), entry_datetime, float(weight_value))
                    
                    # Reload data
                    load_data_files()
                    st.success(f"Weight entry added: {weight_value} lbs on {entry_datetime.strftime('%Y-%m-%d %H:%M')}")
                    st.rerun()
                else:
                    st.error("Please enter a valid weight value")
    
    with tab2:
        st.write("**Add LBM Entry**")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Date and time input (default to today, 9:00 AM)
            lbm_date = st.date_input("Date", value=date.today(), key="lbm_date")
            lbm_time = st.time_input("Time", value=datetime.strptime("09:00", "%H:%M").time(), key="lbm_time")
            lbm_datetime = datetime.combine(lbm_date, lbm_time)
        
        with col2:
            lbm_value = st.number_input("LBM (lbs)", min_value=50.0, max_value=500.0, value=150.0, step=0.1, key="lbm_value")
            
            if st.button("Add LBM Entry", type="primary", key="add_lbm"):
                if lbm_value > 0:
                    # Save to DB (per-user)
                    user_id = get_current_user()
                    if user_id:
                        insert_lbm_for_user(sanitize_user_id(user_id), lbm_datetime, float(lbm_value))
                    
                    # Reload data
                    load_data_files()
                    st.success(f"LBM entry added: {lbm_value} lbs on {lbm_datetime.strftime('%Y-%m-%d %H:%M')}")
                    st.rerun()
                else:
                    st.error("Please enter a valid LBM value")
    
    with tab3:
        st.write("**Add Body Fat % Entry**")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Date and time input (default to today, 9:00 AM)
            bf_date = st.date_input("Date", value=date.today(), key="bf_date")
            bf_time = st.time_input("Time", value=datetime.strptime("09:00", "%H:%M").time(), key="bf_time")
            bf_datetime = datetime.combine(bf_date, bf_time)
        
        with col2:
            bf_value = st.number_input("Body Fat %", min_value=0.0, max_value=100.0, value=15.0, step=0.1, key="bf_value")
            
            if st.button("Add Body Fat % Entry", type="primary", key="add_bf"):
                if 0 <= bf_value <= 100:
                    # Calculate LBM from current weight estimate and body fat %
                    if st.session_state.weights_data:
                        try:
                            # Get current weight estimate using Kalman filter
                            kalman_states, kalman_dates = run_kalman_smoother(st.session_state.weights_data)
                            if kalman_states:
                                current_weight = kalman_states[-1].weight
                                
                                # Calculate LBM: LBM = weight * (1 - body_fat_percent/100)
                                lbm = current_weight * (1.0 - bf_value / 100.0)
                                
                                # Save to DB (per-user)
                                user_id = get_current_user()
                                if user_id:
                                    insert_lbm_for_user(sanitize_user_id(user_id), bf_datetime, float(lbm))
                                
                                # Reload data
                                load_data_files()
                                st.success(f"Body fat entry added: {bf_value}% (LBM: {lbm:.1f} lbs) on {bf_datetime.strftime('%Y-%m-%d %H:%M')}")
                                st.rerun()
                            else:
                                st.error("Failed to get current weight estimate. Please add weight entries first.")
                        except Exception as e:
                            st.error(f"Error calculating LBM: {e}")
                    else:
                        st.error("No weight data available. Please add weight entries first.")
                else:
                    st.error("Body fat percentage must be between 0 and 100")

def show_weight_tracking():
    """Weight tracking page with detailed analysis"""
    from datetime import timedelta
    st.header("📈 Weight Analysis")
    
    # Display current data
    if st.session_state.weights_data:
        st.subheader("📊 Weight Analysis")
        
        # Kalman filter analysis (primary method)
        try:
            kalman_states, kalman_dates = run_kalman_smoother(st.session_state.weights_data)
            
            if kalman_states:
                latest_kalman = kalman_states[-1]
                
                col1, col2, col3 = st.columns(3)
                
                # Get confidence multiplier for error reporting
                ci_mult = get_confidence_multiplier(st.session_state.confidence_interval)
                
                with col1:
                    st.metric(
                        "Smoothed Weight Estimate",
                        f"{latest_kalman.weight:.2f} lbs"
                    )
                    st.caption(f"±{ci_mult * latest_kalman.weight_var**0.5:.2f}")
                
                with col2:
                    velocity_per_week = 7 * latest_kalman.velocity
                    st.metric(
                        "Weekly Rate",
                        f"{velocity_per_week:+.3f} lbs/week"
                    )
                    st.caption(f"±{ci_mult * 7 * (latest_kalman.velocity_var**0.5):.3f}")
                
                with col3:
                    calorie_deficit = latest_kalman.velocity * 3500
                    st.metric(
                        "Calorie Surplus/Deficit",
                        f"{calorie_deficit:+.0f} cal/day"
                    )
                    st.caption("Estimated")
                
                # Forecast metrics (only if enabled)
                if st.session_state.enable_forecast:
                    st.subheader("🔮 Forecast Summary")
                    col1, col2 = st.columns(2)
                    
                    # Calculate forecasts
                    kf = WeightKalmanFilter(
                        initial_weight=latest_kalman.weight,
                        initial_velocity=latest_kalman.velocity,
                        initial_weight_var=latest_kalman.weight_var,
                        initial_velocity_var=latest_kalman.velocity_var,
                    )
                    kf.x = np.array([latest_kalman.weight, latest_kalman.velocity], dtype=float)
                    kf.P = np.array([
                        [latest_kalman.weight_var, latest_kalman.weight_velocity_cov],
                        [latest_kalman.weight_velocity_cov, latest_kalman.velocity_var],
                    ], dtype=float)
                    
                    ci_mult = get_confidence_multiplier(st.session_state.confidence_interval)
                    
                    with col1:
                        week_forecast, week_std = kf.forecast(7.0)
                        st.metric(
                            "1-Week Forecast",
                            f"{week_forecast:.2f} lbs"
                        )
                        st.caption(f"±{ci_mult * week_std:.2f}")
                    
                    with col2:
                        forecast_days = st.session_state.forecast_days
                        forecast_value, forecast_std = kf.forecast(float(forecast_days))
                        st.metric(
                            f"{forecast_days}-Day Forecast",
                            f"{forecast_value:.2f} lbs"
                        )
                        st.caption(f"±{ci_mult * forecast_std:.2f}")
                
                # Smoothed trend analysis
                st.subheader("🔬 Trend Analysis")
                
                # Create dense, smooth curves using the same approach as the original
                from kalman import compute_kalman_mean_std_spline
                dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(kalman_states, kalman_dates)
                
                # Add forecast extension if enabled
                if st.session_state.enable_forecast:
                    # Calculate forecasts using the same approach as dashboard
                    kf = WeightKalmanFilter(
                        initial_weight=latest_kalman.weight,
                        initial_velocity=latest_kalman.velocity,
                        initial_weight_var=latest_kalman.weight_var,
                        initial_velocity_var=latest_kalman.velocity_var,
                    )
                    kf.x = np.array([latest_kalman.weight, latest_kalman.velocity], dtype=float)
                    kf.P = np.array([
                        [latest_kalman.weight_var, latest_kalman.weight_velocity_cov],
                        [latest_kalman.weight_velocity_cov, latest_kalman.velocity_var],
                    ], dtype=float)
                    
                    # Generate forecast data using configurable duration
                    forecast_days = int(st.session_state.forecast_days)
                    latest_date = ensure_py_datetime(dense_datetimes[-1])
                    forecast_dates = [latest_date + timedelta(days=int(i)) for i in range(1, forecast_days + 1)]
                    forecast_weights = []
                    forecast_uncertainties = []
                    
                    for i in range(1, forecast_days + 1):
                        weight, std = kf.forecast(float(i))
                        forecast_weights.append(weight)
                        forecast_uncertainties.append(std)
                    
                    # Combine historical and forecast data
                    combined_dates = list(dense_datetimes) + list(forecast_dates)
                    combined_means = list(dense_means) + list(forecast_weights)
                    combined_stds = list(dense_stds) + list(forecast_uncertainties)
                else:
                    # No forecast, just use historical data
                    combined_dates = dense_datetimes
                    combined_means = dense_means
                    combined_stds = dense_stds
                
                # Create single plot (matching dashboard style)
                fig = go.Figure()
                
                # Add original data points
                dates = [entry.entry_datetime for entry in st.session_state.weights_data]
                weights = [entry.weight for entry in st.session_state.weights_data]
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=weights,
                    mode='markers',
                    name='Weight Measurements',
                    marker=dict(size=8, color='blue', opacity=0.7)
                ))
                
                # Add dense smoothed estimate curve (historical portion)
                fig.add_trace(go.Scatter(
                    x=dense_datetimes,
                    y=dense_means,
                    mode='lines',
                    name='Smoothed Trend',
                    line=dict(color='red', width=2)
                ))
                
                # Add forecast portion with dashed line and fading opacity
                if st.session_state.enable_forecast:
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_weights,
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', width=2, dash='dash'),
                        opacity=0.7  # Overall opacity for the trace
                    ))
                
                # Add smooth confidence intervals (historical portion)
                ci_multiplier = get_confidence_multiplier(st.session_state.confidence_interval)
                hist_upper_bound = [mean + ci_multiplier * std for mean, std in zip(dense_means, dense_stds)]
                hist_lower_bound = [mean - ci_multiplier * std for mean, std in zip(dense_means, dense_stds)]
                
                fig.add_trace(go.Scatter(
                    x=dense_datetimes,
                    y=hist_upper_bound,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=dense_datetimes,
                    y=hist_lower_bound,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    name=f'{st.session_state.confidence_interval} Confidence',
                    hoverinfo='skip'
                ))
                
                # Add forecast confidence intervals with fading opacity
                if st.session_state.enable_forecast:
                    forecast_upper_bound = [mean + ci_multiplier * std for mean, std in zip(forecast_weights, forecast_uncertainties)]
                    forecast_lower_bound = [mean - ci_multiplier * std for mean, std in zip(forecast_weights, forecast_uncertainties)]
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_upper_bound,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip',
                        opacity=0.5
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_lower_bound,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.1)',
                        name=f'Forecast {st.session_state.confidence_interval} CI',
                        hoverinfo='skip',
                        opacity=0.5
                    ))
                
                
                # Set default plot range based on user preference
                start_date, end_date = get_default_plot_range(st.session_state.weights_data)
                layout_config = get_mobile_friendly_layout_config()
                if start_date is not None and end_date is not None:
                    layout_config['xaxis'] = {'range': [start_date, end_date]}
                
                # Set appropriate Y-axis range for weight data
                y_range = get_weight_yaxis_range(st.session_state.weights_data, start_date, end_date)
                if y_range:
                    layout_config['yaxis'] = {'range': y_range}
                
                fig.update_layout(
                    title="Weight Trend Analysis",
                    height=500,
                    hovermode='x unified',
                    **layout_config
                )
                if start_date is not None and end_date is not None:
                    fig.update_xaxes(range=[start_date, end_date])
                if y_range:
                    fig.update_yaxes(range=y_range)
                
                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text="Weight (lbs)")
                
                st.plotly_chart(fig, width='stretch')
                
                # Velocity plot with dense sampling
                st.subheader("📈 Velocity Analysis")
                
                # Create dense velocity curves
                velocities = [state.velocity for state in kalman_states]
                velocity_stds = [state.velocity_var**0.5 for state in kalman_states]
                
                # Use the same dense time grid as the mean/std spline
                from scipy.interpolate import CubicSpline
                t0 = kalman_dates[0]
                t_entry_days = np.array([(d - t0).total_seconds() / 86400.0 for d in kalman_dates], dtype=float)
                velocity_vals = np.array(velocities, dtype=float)
                velocity_std_vals = np.array(velocity_stds, dtype=float)
                
                # Create dense time grid (same as in compute_kalman_mean_std_spline)
                min_t = float(np.min(t_entry_days))
                max_t = float(np.max(t_entry_days))
                dense_t = np.linspace(min_t, max_t, 10000)
                
                # Interpolate velocity and its uncertainty
                try:
                    velocity_spline = CubicSpline(t_entry_days, velocity_vals, bc_type="natural")
                    velocity_std_spline = CubicSpline(t_entry_days, velocity_std_vals, bc_type="natural")
                    dense_velocities = velocity_spline(dense_t)
                    dense_velocity_stds = velocity_std_spline(dense_t)
                except Exception:
                    # Fallback to linear interpolation
                    dense_velocities = np.interp(dense_t, t_entry_days, velocity_vals)
                    dense_velocity_stds = np.interp(dense_t, t_entry_days, velocity_std_vals)
                
                # Convert back to datetime
                dense_velocity_datetimes = [t0 + timedelta(days=float(td)) for td in dense_t]
                
                # Convert velocity from lbs/day to lbs/week
                dense_velocities_weekly = [v * 7 for v in dense_velocities]
                
                # Create velocity plot
                velocity_fig = go.Figure()
                
                velocity_fig.add_trace(go.Scatter(
                    x=dense_velocity_datetimes,
                    y=dense_velocities_weekly,
                    mode='lines',
                    name='Velocity',
                    line=dict(color='green', width=2)
                ))
                
                # Add velocity confidence interval (convert to lbs/week)
                velocity_upper = [(v + ci_multiplier * std) * 7 for v, std in zip(dense_velocities, dense_velocity_stds)]
                velocity_lower = [(v - ci_multiplier * std) * 7 for v, std in zip(dense_velocities, dense_velocity_stds)]
                
                velocity_fig.add_trace(go.Scatter(
                    x=dense_velocity_datetimes,
                    y=velocity_upper,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                velocity_fig.add_trace(go.Scatter(
                    x=dense_velocity_datetimes,
                    y=velocity_lower,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(0,255,0,0.2)',
                    line=dict(width=0),
                    name=f'Velocity {st.session_state.confidence_interval} CI',
                    hoverinfo='skip'
                ))
                
                # Add zero line for velocity
                velocity_fig.add_hline(y=0, line_dash="dash", line_color="gray")
                
                # Set default plot range based on user preference
                start_date, end_date = get_default_plot_range(st.session_state.weights_data)
                layout_config = get_velocity_mobile_layout_config()
                if start_date is not None and end_date is not None:
                    layout_config['xaxis'] = {'range': [start_date, end_date]}
                layout_config['yaxis'] = {'range': [-3, 3]}  # Set y-range to -3 to +3 lbs/week
                
                velocity_fig.update_layout(
                    title="Weight Change Velocity",
                    height=400,
                    hovermode='x unified',
                    **layout_config
                )
                if start_date is not None and end_date is not None:
                    velocity_fig.update_xaxes(range=[start_date, end_date])
                
                velocity_fig.update_xaxes(title_text="Date")
                velocity_fig.update_yaxes(title_text="Velocity (lbs/week)")
                
                st.plotly_chart(velocity_fig, width='stretch')
                
                # Residuals analysis
                st.subheader("📊 Residuals Analysis")
                st.write("Analysis of differences between raw weight measurements and smoothed trend estimates")
                
                # Compute residuals
                from kalman import compute_residuals
                residuals = compute_residuals(st.session_state.weights_data, kalman_states, kalman_dates)
                
                if residuals:
                    residuals_array = np.array(residuals)
                    mean_residual = np.mean(residuals_array)
                    std_residual = np.std(residuals_array, ddof=1)  # Sample standard deviation
                    
                    # Create residuals histogram using Plotly
                    import plotly.figure_factory as ff
                    from scipy import stats
                    
                    # Create histogram data
                    hist_data = [residuals_array]
                    group_labels = [f'Residuals (n={len(residuals_array)})']
                    
                    # Create histogram with user-configurable number of bins
                    data_range = residuals_array.max() - residuals_array.min()
                    bin_size = data_range / st.session_state.residuals_bins
                    
                    fig_hist = ff.create_distplot(
                        hist_data, 
                        group_labels, 
                        bin_size=bin_size,
                        show_curve=True,
                        show_rug=False
                    )
                    
                    # Add normal distribution overlay
                    x_range = np.linspace(residuals_array.min(), residuals_array.max(), 100)
                    normal_dist = stats.norm.pdf(x_range, loc=0, scale=std_residual)
                    
                    fig_hist.add_trace(go.Scatter(
                        x=x_range,
                        y=normal_dist,
                        mode='lines',
                        name=f'Normal(μ=0, σ={std_residual:.3f})',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Add vertical lines for statistics
                    ci_mult = get_confidence_multiplier(st.session_state.confidence_interval)
                    
                    # Add traces for legend entries (mobile-friendly, no duplicate lines)
                    y_max = max(normal_dist) * 1.1
                    y_min = 0  # Don't extend below zero
                    
                    fig_hist.add_trace(go.Scatter(
                        x=[mean_residual, mean_residual],
                        y=[y_min, y_max],
                        mode='lines',
                        line=dict(color='green', dash='dash'),
                        name=f"Mean = {mean_residual:.3f}",
                        showlegend=True,
                        hoverinfo='skip'
                    ))
                    
                    fig_hist.add_trace(go.Scatter(
                        x=[ci_mult * std_residual, ci_mult * std_residual],
                        y=[y_min, y_max],
                        mode='lines',
                        line=dict(color='red', dash='dot'),
                        name=f"+{ci_mult:.1f}σ = {ci_mult * std_residual:.3f}",
                        showlegend=True,
                        hoverinfo='skip'
                    ))
                    
                    fig_hist.add_trace(go.Scatter(
                        x=[-ci_mult * std_residual, -ci_mult * std_residual],
                        y=[y_min, y_max],
                        mode='lines',
                        line=dict(color='red', dash='dot'),
                        name=f"-{ci_mult:.1f}σ = {-ci_mult * std_residual:.3f}",
                        showlegend=True,
                        hoverinfo='skip'
                    ))
                    
                    fig_hist.update_layout(
                        title="Residuals Histogram (Smoothed vs Raw Data)",
                        xaxis_title="Residual (Raw Weight - Smoothed Weight) [lbs]",
                        yaxis_title="Density",
                        height=500,
                        showlegend=True,
                        **get_residuals_mobile_layout_config()
                    )
                    
                    st.plotly_chart(fig_hist, width='stretch')
                    
                    # Statistics summary
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean Residual", f"{mean_residual:.4f} lbs")
                    
                    with col2:
                        st.metric("Std Deviation", f"{std_residual:.4f} lbs")
                    
                    with col3:
                        # Calculate skewness and kurtosis
                        skewness = stats.skew(residuals_array)
                        excess_kurtosis = stats.kurtosis(residuals_array)
                        st.metric("Skewness", f"{skewness:.4f}")
                    
                    # Normality test
                    if len(residuals_array) >= 3:
                        shapiro_stat, shapiro_p = stats.shapiro(residuals_array)
                        ks_stat, ks_p = stats.kstest(residuals_array, 'norm', args=(mean_residual, std_residual))
                        
                        st.info(f"""
                        **Normality Tests:**
                        - Shapiro-Wilk: p = {shapiro_p:.4f} {'(Normal)' if shapiro_p > 0.05 else '(Not Normal)'}
                        - Kolmogorov-Smirnov: p = {ks_p:.4f} {'(Normal)' if ks_p > 0.05 else '(Not Normal)'}
                        
                        *p > 0.05: Residuals appear normal | p ≤ 0.05: Residuals may not be normal*
                        """)
                else:
                    st.warning("No residuals data available for analysis")
        
        except Exception as e:
            st.error(f"Error running trend analysis: {e}")
    
    else:
        st.info("No weight data available. Please add some entries above.")

def show_body_composition():
    """Body composition analysis page"""
    st.header("📊 Composition Analysis Analysis")
    
    if not st.session_state.weights_data:
        st.warning("No weight data available. Please add some weight entries first.")
        return
    
    # Height input
    st.subheader("📏 Height Settings")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        height_value = st.number_input("Height", min_value=48.0, max_value=96.0, value=st.session_state.height, step=0.1)
    
    with col2:
        height_unit = st.selectbox("Unit", ["inches", "cm"])
        if height_unit == "cm":
            height_inches = height_value * 0.393701
        else:
            height_inches = height_value
    
    if st.button("Update Height"):
        save_height(height_inches)
        st.success(f"Height updated to {height_value} {height_unit} ({height_inches:.1f} inches)")
        st.rerun()
    
    # BMI calculation
    st.subheader("📊 BMI Analysis")
    
    try:
        kalman_states, kalman_dates = run_kalman_smoother(st.session_state.weights_data)
        
        if kalman_states:
            # Calculate BMI for each Kalman state
            bmi_values = []
            for state in kalman_states:
                bmi = (state.weight * 703) / (height_inches ** 2)
                bmi_values.append(bmi)
            
            # Display current BMI
            current_bmi = bmi_values[-1]
            bmi_category = get_bmi_category(current_bmi)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current BMI", f"{current_bmi:.1f}")
            
            with col2:
                st.metric("BMI Category", bmi_category)
            
            with col3:
                st.metric("Height", f"{height_inches:.1f} inches")
            
            # BMI trend chart with dense Kalman sampling
            from kalman import compute_kalman_mean_std_spline
            dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(kalman_states, kalman_dates)
            
            # Calculate BMI for dense Kalman data
            dense_bmi_values = []
            for weight in dense_means:
                bmi = (weight * 703) / (height_inches ** 2)
                dense_bmi_values.append(bmi)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dense_datetimes,
                y=dense_bmi_values,
                mode='lines',
                name='BMI (Kalman)',
                line=dict(color='blue', width=2)
            ))
            
            # Add BMI category lines
            fig.add_hline(y=18.5, line_dash="dash", line_color="green", annotation_text="Underweight")
            fig.add_hline(y=25, line_dash="dash", line_color="yellow", annotation_text="Normal")
            fig.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Overweight")
            fig.add_hline(y=35, line_dash="dash", line_color="red", annotation_text="Obese")
            
            # Set default plot range based on user preference
            start_date, end_date = get_default_plot_range(st.session_state.weights_data)
            layout_config = get_mobile_friendly_layout_config()
            if start_date is not None and end_date is not None:
                layout_config['xaxis'] = {'range': [start_date, end_date]}
            
            fig.update_layout(
                title="BMI Trend Over Time",
                xaxis_title="Date",
                yaxis_title="BMI",
                height=500,
                **layout_config
            )
            if start_date is not None and end_date is not None:
                fig.update_xaxes(range=[start_date, end_date])
            
            st.plotly_chart(fig, width='stretch')
    
    except Exception as e:
        st.error(f"Error calculating BMI: {e}")
    
    # Body fat analysis (if LBM data available)
    if not st.session_state.lbm_data.empty:
        st.subheader("🩸 Body Fat Analysis")
        
        try:
            # Use Kalman filter approach for body fat analysis
            kalman_states, kalman_dates = run_kalman_smoother(st.session_state.weights_data)
            
            if kalman_states:
                # Create dense, smooth curves using Kalman filter
                from kalman import compute_kalman_mean_std_spline
                dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(kalman_states, kalman_dates)
                
                # Process LBM data
                lbm_dates = pd.to_datetime(st.session_state.lbm_data['date'], format='ISO8601')
                lbm_values = st.session_state.lbm_data['lbm'].values
                
                # Interpolate LBM to dense Kalman dates
                from scipy.interpolate import interp1d
                
                # Convert LBM dates to seconds from first date
                lbm_times = (lbm_dates - lbm_dates[0]).dt.total_seconds().values
                
                # Create interpolation function that holds last value constant beyond data range
                lbm_interp = interp1d(
                    lbm_times,
                    lbm_values,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(lbm_values[0], lbm_values[-1])  # Hold first/last values constant
                )
                
                # Calculate body fat percentages for dense Kalman data
                bf_percentages = []
                lbm_interpolated = []
                last_lbm_date = lbm_dates.iloc[-1]
                
                for i, (dt, weight) in enumerate(zip(dense_datetimes, dense_means)):
                    time_diff = (dt - dense_datetimes[0]).total_seconds()
                    lbm = lbm_interp(time_diff)
                    lbm_interpolated.append(lbm)
                    bf_percent = ((weight - lbm) / weight) * 100
                    bf_percentages.append(bf_percent)
                
                if bf_percentages:
                    # Display current body fat
                    current_bf = bf_percentages[-1]
                    current_lbm = lbm_interpolated[-1]
                    bf_category = get_bf_category(current_bf, 'male')  # Assuming male for now
                    
                    # Check if current LBM is being held constant (beyond last measurement)
                    current_date = dense_datetimes[-1]
                    is_lbm_held_constant = current_date > last_lbm_date
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Body Fat %", f"{current_bf:.1f}%")
                    
                    with col2:
                        st.metric("Body Fat Category", bf_category)
                    
                    with col3:
                        st.metric("Current LBM", f"{current_lbm:.1f} lbs")
                    
                    # Show note if LBM is being held constant
                    if is_lbm_held_constant:
                        st.info(f"💡 **LBM Note**: Your last LBM measurement was on {last_lbm_date.strftime('%Y-%m-%d')}. Current LBM estimate assumes your lean mass has remained constant since then.")
                    
                    # Body fat trend chart with Kalman smoothing (lines only)
                    fig = go.Figure()
                    
                    # Add dense Kalman-smoothed body fat curve
                    fig.add_trace(go.Scatter(
                        x=dense_datetimes,
                        y=bf_percentages,
                        mode='lines',
                        name='Body Fat % (Smoothed)',
                        line=dict(color='purple', width=2)
                    ))
                    
                    # Set default plot range based on user preference
                    start_date, end_date = get_default_plot_range(st.session_state.weights_data)
                    layout_config = get_mobile_friendly_layout_config()
                    if start_date is not None and end_date is not None:
                        layout_config['xaxis'] = {'range': [start_date, end_date]}
                    
                    fig.update_layout(
                        title="Body Fat Percentage Trend (Smoothed)",
                        xaxis_title="Date",
                        yaxis_title="Body Fat %",
                        height=500,
                        **layout_config
                    )
                    if start_date is not None and end_date is not None:
                        fig.update_xaxes(range=[start_date, end_date])
                    
                    st.plotly_chart(fig, width='stretch')
                
                # FFMI Analysis
                st.subheader("💪 FFMI Analysis")
                
                # Calculate FFMI (Fat-Free Mass Index) using Kalman-smoothed data
                ffmi_values = []
                for lbm in lbm_interpolated:
                    # FFMI = LBM (kg) / height (m)²
                    lbm_kg = lbm * 0.453592  # Convert lbs to kg
                    height_m = height_inches * 0.0254  # Convert inches to meters
                    ffmi = lbm_kg / (height_m ** 2)
                    ffmi_values.append(ffmi)
                
                current_ffmi = ffmi_values[-1]
                ffmi_category = get_ffmi_category(current_ffmi, 'male')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Current FFMI", f"{current_ffmi:.1f}")
                
                with col2:
                    st.metric("FFMI Category", ffmi_category)
                
                # FFMI trend chart (lines only, no markers)
                ffmi_fig = go.Figure()
                
                ffmi_fig.add_trace(go.Scatter(
                    x=dense_datetimes,
                    y=ffmi_values,
                    mode='lines',
                    name='FFMI (Smoothed)',
                    line=dict(color='green', width=2)
                ))
                
                # Add FFMI reference lines
                ffmi_fig.add_hline(y=16, line_dash="dash", line_color="red", annotation_text="Below Average")
                ffmi_fig.add_hline(y=18, line_dash="dash", line_color="orange", annotation_text="Average")
                ffmi_fig.add_hline(y=20, line_dash="dash", line_color="yellow", annotation_text="Above Average")
                ffmi_fig.add_hline(y=22, line_dash="dash", line_color="green", annotation_text="Excellent")
                
                # Set default plot range based on user preference
                start_date, end_date = get_default_plot_range(st.session_state.weights_data)
                layout_config = get_mobile_friendly_layout_config()
                if start_date is not None and end_date is not None:
                    layout_config['xaxis'] = {'range': [start_date, end_date]}
                
                ffmi_fig.update_layout(
                    title="Fat-Free Mass Index (FFMI) Trend (Smoothed)",
                    xaxis_title="Date",
                    yaxis_title="FFMI",
                    height=500,
                    **layout_config
                )
                if start_date is not None and end_date is not None:
                    ffmi_fig.update_xaxes(range=[start_date, end_date])
                
                st.plotly_chart(ffmi_fig, width='stretch')
        
        except Exception as e:
            st.error(f"Error calculating body fat: {e}")
    
    else:
        st.info("No LBM data available. Upload a CSV file with 'date' and 'lbm' columns to see body fat analysis.")

def show_settings():
    """Settings and configuration page"""
    st.header("⚙️ Settings")
    
    st.subheader("📊 Analysis Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_interval = st.selectbox(
            "Confidence Interval",
            ["1σ", "95%"],
            index=0 if st.session_state.confidence_interval == "1σ" else 1,
            help="Confidence level for uncertainty bands"
        )
        
        # Update session state immediately when changed
        if confidence_interval != st.session_state.confidence_interval:
            st.session_state.confidence_interval = confidence_interval
            try:
                user_id = get_current_user()
                if user_id:
                    set_preferences_for_user(
                        sanitize_user_id(user_id),
                        st.session_state.confidence_interval,
                        bool(st.session_state.enable_forecast),
                        int(st.session_state.forecast_days),
                        int(st.session_state.residuals_bins),
                        int(st.session_state.default_plot_range_days),
                    )
            except Exception:
                pass
            st.rerun()
    
    with col2:
        st.info("**Note**: This app uses optimized Kalman filtering with carefully tuned parameters for best results.")
    
    st.subheader("📊 Plot Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        aggregation_hours = st.slider(
            "Aggregation Window (hours)",
            min_value=0,
            max_value=24,
            value=3,
            help="Group measurements within this window (0 to disable)"
        )
        
        enable_forecast = st.checkbox(
            "Enable Forecasting",
            value=st.session_state.enable_forecast,
            help="Show forecast projections on plots"
        )
        
        forecast_days = st.slider(
            "Forecast Duration (days)",
            min_value=7,
            max_value=90,
            value=st.session_state.forecast_days,
            help="Number of days to forecast into the future"
        )
        
        default_plot_range_days = st.slider(
            "Default Plot Range (days)",
            min_value=7,
            max_value=365,
            value=st.session_state.default_plot_range_days,
            help="Default time range shown on plots (you can still zoom out to see all data)"
        )
        
        # Y-axis padding setting
        if 'yaxis_padding_factor' not in st.session_state:
            st.session_state.yaxis_padding_factor = 0.1
        
        yaxis_padding = st.slider(
            "Y-axis Padding (%)",
            min_value=5,
            max_value=25,
            value=int(st.session_state.yaxis_padding_factor * 100),
            help="Extra space above and below weight data on charts (5% = tight fit, 25% = lots of space)"
        )
        st.session_state.yaxis_padding_factor = yaxis_padding / 100.0
        
        # Update session state immediately when changed
        if enable_forecast != st.session_state.enable_forecast:
            st.session_state.enable_forecast = enable_forecast
            try:
                user_id = get_current_user()
                if user_id:
                    set_preferences_for_user(
                        sanitize_user_id(user_id),
                        st.session_state.confidence_interval,
                        bool(st.session_state.enable_forecast),
                        int(st.session_state.forecast_days),
                        int(st.session_state.residuals_bins),
                        int(st.session_state.default_plot_range_days),
                    )
            except Exception:
                pass
            st.rerun()
        
        if forecast_days != st.session_state.forecast_days:
            st.session_state.forecast_days = forecast_days
            try:
                user_id = get_current_user()
                if user_id:
                    set_preferences_for_user(
                        sanitize_user_id(user_id),
                        st.session_state.confidence_interval,
                        bool(st.session_state.enable_forecast),
                        int(st.session_state.forecast_days),
                        int(st.session_state.residuals_bins),
                        int(st.session_state.default_plot_range_days),
                    )
            except Exception:
                pass
            st.rerun()
    
    with col2:
        residuals_bins = st.slider(
            "Residuals Histogram Bins",
            min_value=5,
            max_value=50,
            value=st.session_state.residuals_bins,
            help="Number of bins for the residuals histogram (more bins = more detail)"
        )
        
        # Update session state immediately when changed
        if residuals_bins != st.session_state.residuals_bins:
            st.session_state.residuals_bins = residuals_bins
            try:
                user_id = get_current_user()
                if user_id:
                    set_preferences_for_user(
                        sanitize_user_id(user_id),
                        st.session_state.confidence_interval,
                        bool(st.session_state.enable_forecast),
                        int(st.session_state.forecast_days),
                        int(st.session_state.residuals_bins),
                        int(st.session_state.default_plot_range_days),
                    )
            except Exception:
                pass
            st.rerun()
        
        # Update session state immediately when changed
        if default_plot_range_days != st.session_state.default_plot_range_days:
            st.session_state.default_plot_range_days = default_plot_range_days
            try:
                user_id = get_current_user()
                if user_id:
                    set_preferences_for_user(
                        sanitize_user_id(user_id),
                        st.session_state.confidence_interval,
                        bool(st.session_state.enable_forecast),
                        int(st.session_state.forecast_days),
                        int(st.session_state.residuals_bins),
                        int(st.session_state.default_plot_range_days),
                    )
            except Exception:
                pass
            st.rerun()
        
    
    if st.button("Save Settings", type="primary"):
        # Settings are already saved automatically when changed
        st.success("Settings are automatically applied when changed!")
    
    st.subheader("📁 Data Export")
    
    if st.session_state.weights_data:
        # Export weights data
        weights_df = pd.DataFrame([
            {
                'date': entry.entry_datetime.isoformat(),
                'weight': entry.weight
            }
            for entry in st.session_state.weights_data
        ])
        
        csv_weights = weights_df.to_csv(index=False)
        st.download_button(
            label="Download Weights CSV",
            data=csv_weights,
            file_name=f"weights_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    if not st.session_state.lbm_data.empty:
        csv_lbm = st.session_state.lbm_data.to_csv(index=False)
        st.download_button(
            label="Download LBM CSV",
            data=csv_lbm,
            file_name=f"lbm_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def show_data_management():
    """Data management page for uploading and managing CSV files"""
    st.header("📁 Data Management")
    
    # File upload section
    st.subheader("📤 Upload Data Files")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Weights CSV**")
        weights_file = st.file_uploader(
            "Upload weights CSV",
            type=['csv'],
            help="CSV should have 'date' and 'weight' columns"
        )
        
        if weights_file is not None:
            try:
                df = pd.read_csv(weights_file)
                if 'date' in df.columns and 'weight' in df.columns:
                    # Replace user's weights in DB
                    user_id = get_current_user()
                    if user_id:
                        replace_weights_for_user(sanitize_user_id(user_id), df)
                    
                    # Reload data
                    load_data_files()
                    st.success(f"Successfully uploaded {len(df)} weight entries")
                    st.rerun()
                else:
                    st.error("CSV must have 'date' and 'weight' columns")
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    with col2:
        st.write("**LBM CSV**")
        lbm_file = st.file_uploader(
            "Upload LBM CSV",
            type=['csv'],
            help="CSV should have 'date' and 'lbm' columns"
        )
        
        if lbm_file is not None:
            try:
                # Try to read the CSV file
                df = pd.read_csv(lbm_file)
                
                # Check if we have the required columns
                if 'date' in df.columns and 'lbm' in df.columns:
                    # Validate data types
                    try:
                        df['lbm'] = pd.to_numeric(df['lbm'])
                        df['date'] = pd.to_datetime(df['date'])
                    except Exception as e:
                        st.error(f"Error converting data types: {e}")
                        return
                    
                    # Replace user's LBM data in DB
                    user_id = get_current_user()
                    if user_id:
                        replace_lbm_for_user(sanitize_user_id(user_id), df)
                    
                    # Reload data
                    load_data_files()
                    st.success(f"Successfully uploaded {len(df)} LBM entries")
                    st.rerun()
                else:
                    # Try to handle files without proper headers
                    st.warning("CSV doesn't have 'date' and 'lbm' columns. Attempting to fix...")
                    
                    if len(df.columns) >= 2:
                        # Assume first column is date, second is lbm
                        df_fixed = pd.DataFrame()
                        df_fixed['date'] = df.iloc[:, 0]
                        df_fixed['lbm'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                        
                        # Remove rows with invalid data
                        df_fixed = df_fixed.dropna()
                        
                        if len(df_fixed) > 0:
                            # Convert date column
                            df_fixed['date'] = pd.to_datetime(df_fixed['date'], errors='coerce')
                            df_fixed = df_fixed.dropna()
                            
                            if len(df_fixed) > 0:
                                # Replace user's LBM data in DB (auto-fixed format)
                                user_id = get_current_user()
                                if user_id:
                                    replace_lbm_for_user(sanitize_user_id(user_id), df_fixed)
                                
                                # Reload data
                                load_data_files()
                                st.success(f"Successfully uploaded {len(df_fixed)} LBM entries (auto-fixed format)")
                                st.rerun()
                            else:
                                st.error("Could not parse dates in the CSV file")
                        else:
                            st.error("Could not parse LBM values in the CSV file")
                    else:
                        st.error("CSV must have at least 2 columns (date and lbm)")
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Please ensure your CSV has 'date' and 'lbm' columns, or at least 2 columns with date in first column and LBM values in second column.")
    
    with col3:
        st.write("**FITINDEX CSV**")
        fitindex_file = st.file_uploader(
            "Upload FITINDEX CSV",
            type=['csv'],
            help="FITINDEX app export CSV file"
        )
        
        if fitindex_file is not None:
            try:
                df = pd.read_csv(fitindex_file)
                
                # Check if this looks like a FITINDEX file
                if 'Time of Measurement' in df.columns and 'Weight(lb)' in df.columns:
                    st.success("FITINDEX file detected! Converting to standard format...")
                    
                    # Convert FITINDEX data to our standard format
                    converted_entries = []
                    
                    for _, row in df.iterrows():
                        try:
                            # Parse the datetime from FITINDEX format
                            time_str = str(row['Time of Measurement']).strip()
                            weight = float(row['Weight(lb)'])
                            
                            # Parse the datetime (format: "MM/DD/YYYY, HH:MM:SS")
                            dt = datetime.strptime(time_str, "%m/%d/%Y, %H:%M:%S")
                            
                            # Create WeightEntry
                            entry = WeightEntry(dt, weight)
                            converted_entries.append(entry)
                            
                        except (ValueError, TypeError) as e:
                            # Skip invalid entries
                            continue
                    
                    if converted_entries:
                        # Sort entries by datetime
                        converted_entries.sort(key=lambda x: x.entry_datetime)
                        
                        # Replace user's weights in DB
                        user_id = get_current_user()
                        if user_id:
                            conv_df = pd.DataFrame({
                                'date': [e.entry_datetime for e in converted_entries],
                                'weight': [e.weight for e in converted_entries]
                            })
                            replace_weights_for_user(sanitize_user_id(user_id), conv_df)
                        
                        # Reload data
                        load_data_files()
                        st.success(f"Successfully converted and uploaded {len(converted_entries)} weight entries from FITINDEX data")
                        st.info("FITINDEX data has been converted to the standard weights format and is now your primary data source.")
                        st.rerun()
                    else:
                        st.error("No valid weight entries found in the FITINDEX file")
                else:
                    st.error("This doesn't appear to be a FITINDEX CSV file. Expected columns: 'Time of Measurement' and 'Weight(lb)'")
            except Exception as e:
                st.error(f"Error processing FITINDEX file: {e}")
    
    # Data summary
    st.subheader("📊 Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Weight Data**")
        if st.session_state.weights_data:
            st.write(f"Entries: {len(st.session_state.weights_data)}")
            st.write(f"Date range: {st.session_state.weights_data[0].entry_datetime.date()} to {st.session_state.weights_data[-1].entry_datetime.date()}")
            st.write(f"Latest weight: {st.session_state.weights_data[-1].weight:.1f} lbs")
        else:
            st.write("No weight data available")
    
    with col2:
        st.write("**LBM Data**")
        if not st.session_state.lbm_data.empty:
            st.write(f"Entries: {len(st.session_state.lbm_data)}")
            st.write(f"Date range: {st.session_state.lbm_data['date'].min()} to {st.session_state.lbm_data['date'].max()}")
            st.write(f"Latest LBM: {st.session_state.lbm_data['lbm'].iloc[-1]:.1f} lbs")
        else:
            st.write("No LBM data available")
    
    # Data Management Tabs
    st.subheader("📊 Data Management")
    
    tab1, tab2, tab3, tab4 = st.tabs(["View & Edit Data", "Add Manual Entries", "Bulk Operations", "Data Preview"])
    
    with tab1:
        st.write("**View and manage your data entries**")
        
        # Weight Data Management
        if st.session_state.weights_data:
            st.write("**Weight Entries**")
            
            # Create editable dataframe for weights
            weights_data = []
            for i, entry in enumerate(st.session_state.weights_data):
                weights_data.append({
                    'Index': i,
                    'Date': entry.entry_datetime.strftime('%Y-%m-%d'),
                    'Time': entry.entry_datetime.strftime('%H:%M'),
                    'Weight (lbs)': entry.weight,
                    'Delete': False
                })
            
            weights_df = pd.DataFrame(weights_data)
            
            # Display with checkboxes for deletion
            edited_weights_df = st.data_editor(
                weights_df,
                column_config={
                    "Index": st.column_config.NumberColumn("Index", disabled=True),
                    "Date": st.column_config.TextColumn("Date", disabled=True),
                    "Time": st.column_config.TextColumn("Time", disabled=True),
                    "Weight (lbs)": st.column_config.NumberColumn("Weight (lbs)", format="%.1f"),
                    "Delete": st.column_config.CheckboxColumn("Delete", help="Check to delete this entry")
                },
                hide_index=True,
                use_container_width=True,
                key="weights_editor"
            )
            
            # Handle deletions
            if st.button("Delete Selected Weight Entries", type="secondary"):
                user_id = get_current_user()
                if user_id:
                    deleted_count = 0
                    for idx, row in edited_weights_df.iterrows():
                        if row['Delete']:
                            entry = st.session_state.weights_data[row['Index']]
                            if delete_weight_entry(sanitize_user_id(user_id), entry.entry_datetime):
                                deleted_count += 1
                    
                    if deleted_count > 0:
                        load_data_files()
                        st.success(f"Deleted {deleted_count} weight entries")
                        st.rerun()
                    else:
                        st.warning("No entries selected for deletion")
        else:
            st.info("No weight data available")
        
        # LBM Data Management
        if not st.session_state.lbm_data.empty:
            st.write("**LBM Entries**")
            
            # Create editable dataframe for LBM
            lbm_data = []
            for i, (_, row) in enumerate(st.session_state.lbm_data.iterrows()):
                lbm_data.append({
                    'Index': i,
                    'Date': pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
                    'LBM (lbs)': row['lbm'],
                    'Delete': False
                })
            
            lbm_df = pd.DataFrame(lbm_data)
            
            # Display with checkboxes for deletion
            edited_lbm_df = st.data_editor(
                lbm_df,
                column_config={
                    "Index": st.column_config.NumberColumn("Index", disabled=True),
                    "Date": st.column_config.TextColumn("Date", disabled=True),
                    "LBM (lbs)": st.column_config.NumberColumn("LBM (lbs)", format="%.1f"),
                    "Delete": st.column_config.CheckboxColumn("Delete", help="Check to delete this entry")
                },
                hide_index=True,
                use_container_width=True,
                key="lbm_editor"
            )
            
            # Handle deletions
            if st.button("Delete Selected LBM Entries", type="secondary"):
                user_id = get_current_user()
                if user_id:
                    deleted_count = 0
                    for idx, row in edited_lbm_df.iterrows():
                        if row['Delete']:
                            lbm_row = st.session_state.lbm_data.iloc[row['Index']]
                            entry_datetime = pd.to_datetime(lbm_row['date'])
                            if delete_lbm_entry(sanitize_user_id(user_id), entry_datetime):
                                deleted_count += 1
                    
                    if deleted_count > 0:
                        load_data_files()
                        st.success(f"Deleted {deleted_count} LBM entries")
                        st.rerun()
                    else:
                        st.warning("No entries selected for deletion")
        else:
            st.info("No LBM data available")
    
    with tab2:
        st.write("**Add new entries manually**")
        
        # Manual Weight Entry
        st.write("**Add Weight Entry**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            manual_weight_date = st.date_input("Date", value=date.today(), key="manual_weight_date")
        with col2:
            manual_weight_time = st.time_input("Time", value=datetime.strptime("09:00", "%H:%M").time(), key="manual_weight_time")
        with col3:
            manual_weight_value = st.number_input("Weight (lbs)", min_value=50.0, max_value=500.0, value=150.0, step=0.1, key="manual_weight_value")
        
        if st.button("Add Weight Entry", type="primary", key="add_manual_weight"):
            if manual_weight_value > 0:
                entry_datetime = datetime.combine(manual_weight_date, manual_weight_time)
                user_id = get_current_user()
                if user_id:
                    insert_weight_for_user(sanitize_user_id(user_id), entry_datetime, float(manual_weight_value))
                    load_data_files()
                    st.success(f"Weight entry added: {manual_weight_value} lbs on {entry_datetime.strftime('%Y-%m-%d %H:%M')}")
                    st.rerun()
        
        # Manual LBM Entry
        st.write("**Add LBM Entry**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            manual_lbm_date = st.date_input("Date", value=date.today(), key="manual_lbm_date")
        with col2:
            manual_lbm_time = st.time_input("Time", value=datetime.strptime("09:00", "%H:%M").time(), key="manual_lbm_time")
        with col3:
            manual_lbm_value = st.number_input("LBM (lbs)", min_value=50.0, max_value=300.0, value=120.0, step=0.1, key="manual_lbm_value")
        
        if st.button("Add LBM Entry", type="primary", key="add_manual_lbm"):
            if manual_lbm_value > 0:
                entry_datetime = datetime.combine(manual_lbm_date, manual_lbm_time)
                user_id = get_current_user()
                if user_id:
                    # Create DataFrame for LBM entry
                    lbm_df = pd.DataFrame({
                        'date': [entry_datetime],
                        'lbm': [manual_lbm_value]
                    })
                    insert_lbm_for_user(sanitize_user_id(user_id), lbm_df)
                    load_data_files()
                    st.success(f"LBM entry added: {manual_lbm_value} lbs on {entry_datetime.strftime('%Y-%m-%d %H:%M')}")
                    st.rerun()
    
    with tab3:
        st.write("**Bulk operations on your data**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Weight Data Operations**")
            if st.session_state.weights_data:
                if st.button("Clear All Weight Data", type="secondary"):
                    user_id = get_current_user()
                    if user_id:
                        replace_weights_for_user(sanitize_user_id(user_id), pd.DataFrame(columns=['date', 'weight']))
                        load_data_files()
                        st.success("All weight data cleared")
                        st.rerun()
            else:
                st.info("No weight data to clear")
        
        with col2:
            st.write("**LBM Data Operations**")
            if not st.session_state.lbm_data.empty:
                if st.button("Clear All LBM Data", type="secondary"):
                    user_id = get_current_user()
                    if user_id:
                        replace_lbm_for_user(sanitize_user_id(user_id), pd.DataFrame(columns=['date', 'lbm']))
                        load_data_files()
                        st.success("All LBM data cleared")
                        st.rerun()
            else:
                st.info("No LBM data to clear")
    
    with tab4:
        st.write("**Quick data preview**")
        
        # Weight Data Preview
        if st.session_state.weights_data:
            st.write("**Recent Weight Entries**")
            recent_weights = st.session_state.weights_data[-10:]
            weights_df = pd.DataFrame([
                {
                    'Date': entry.entry_datetime.strftime('%Y-%m-%d %H:%M'),
                    'Weight (lbs)': f"{entry.weight:.1f}"
                }
                for entry in recent_weights
            ])
            st.dataframe(weights_df, width='stretch')
        else:
            st.info("No weight data available")
        
        # LBM Data Preview
        if not st.session_state.lbm_data.empty:
            st.write("**Recent LBM Entries**")
            st.dataframe(st.session_state.lbm_data.tail(10), width='stretch')
        else:
            st.info("No LBM data available")

def get_bmi_category(bmi: float) -> str:
    """Get BMI category string"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_bf_category(bf_percent: float, gender: str) -> str:
    """Get body fat category string"""
    if gender.lower() == 'male':
        if bf_percent < 6:
            return "Essential fat"
        elif bf_percent < 14:
            return "Athletes"
        elif bf_percent < 18:
            return "Fitness"
        elif bf_percent < 25:
            return "Average"
        else:
            return "Obese"
    else:  # female
        if bf_percent < 10:
            return "Essential fat"
        elif bf_percent < 16:
            return "Athletes"
        elif bf_percent < 20:
            return "Fitness"
        elif bf_percent < 25:
            return "Average"
        else:
            return "Obese"

def get_ffmi_category(ffmi: float, gender: str) -> str:
    """Get FFMI category string"""
    if gender.lower() == 'male':
        if ffmi < 16:
            return "Below Average"
        elif ffmi < 18:
            return "Average"
        elif ffmi < 20:
            return "Above Average"
        elif ffmi < 22:
            return "Excellent"
        else:
            return "Exceptional"
    else:  # female
        if ffmi < 14:
            return "Below Average"
        elif ffmi < 16:
            return "Average"
        elif ffmi < 18:
            return "Above Average"
        elif ffmi < 20:
            return "Excellent"
        else:
            return "Exceptional"

def show_user_guide():
    """Display comprehensive user guide and documentation"""
    st.header("User Guide")
    
    st.markdown("""
    Welcome to BodyMetrics, a comprehensive fitness tracking application designed to help you monitor and analyze your weight, body composition, and fitness progress over time.
    """)
    
    # Table of Contents
    st.markdown("""
    ## Table of Contents
    
    1. [Getting Started](#getting-started)
    2. [Dashboard Overview](#dashboard-overview)
    3. [Adding Data Entries](#adding-data-entries)
    4. [Weight Analysis](#weight-analysis)
    5. [Body Composition Analysis](#body-composition-analysis)
    6. [Settings and Configuration](#settings-and-configuration)
    7. [Data Management](#data-management)
    8. [Understanding the Analytics](#understanding-the-analytics)
    9. [Troubleshooting](#troubleshooting)
    10. [Best Practices](#best-practices)
    """)
    
    st.markdown("---")
    
    # Getting Started
    st.markdown("""
    ## Getting Started
    
    ### Initial Setup
    1. **Authentication**: Log in with your credentials or create a new account
    2. **Height Configuration**: Set your height in the Settings page (required for BMI calculations)
    3. **First Data Entry**: Add your first weight measurement using the Add Entries page. Alternatively, you can import your data from a CSV file, optionally one from a FITINDEX app export.
    
    ### Navigation
    Use the sidebar navigation to access different features:
    - **Dashboard**: Overview of your current metrics and recent trends
    - **Add Entries**: Input new weight, lean body mass, or body fat measurements
    - **Weight Analysis**: Detailed weight tracking with advanced analytics
    - **Composition Analysis**: Body composition metrics and trends
    - **Settings**: Configure preferences and user information
    - **Data Management**: Import, export, and manage your data files
    - **User Guide**: This comprehensive guide
    - **Disclaimer**: Important legal and health disclaimers
    """)
    
    # Dashboard Overview
    st.markdown("""
    ## Dashboard Overview
    
    The Dashboard provides a high-level view of your fitness progress:
    
    ### Key Metrics
    - **Current Weight**: Your most recent weight measurement
    - **Weight Change**: Net change since your first entry
    - **BMI**: Body Mass Index calculated from your height and current weight
    - **Data Points**: Total number of measurements recorded
    
    ### Visual Elements
    - **Weight Trend Chart**: Interactive plot showing your weight progression over time
    - **Quick Stats**: Summary statistics in the sidebar
    - **Recent Entries**: Latest data points for quick reference
    """)
    
    # Adding Data Entries
    st.markdown("""
    ## Adding Data Entries
    
    ### Weight Entries
    1. Navigate to **Add Entries** page
    2. Select the **Weight Entry** tab
    3. Choose the date and time of measurement
    4. Enter your weight in pounds
    5. Click **Add Weight Entry**
    
    ### Lean Body Mass (LBM) Entries
    1. Select the **LBM Entry** tab
    2. Enter the date and time
    3. Input your lean body mass value
    4. Click **Add LBM Entry**
    
    ### Body Fat Percentage Entries
    1. Select the **Body Fat % Entry** tab
    2. Enter the date and time
    3. Input your body fat percentage
    4. Click **Add Body Fat % Entry**
    
    ### Data Entry Tips
    - **Consistency**: Measure at the same time of day for best results
    - **Accuracy**: Use a reliable scale and ensure proper calibration
    - **Frequency**: Daily measurements provide the most detailed tracking
    - **Units**: All weights are in pounds; ensure your scale is set correctly
    """)
    
    # Weight Analysis
    st.markdown("""
    ## Weight Analysis
    
    The Weight Analysis page provides advanced tracking and forecasting capabilities:
    
    ### Kalman Filter Analysis
    - **Smoothed Trends**: Advanced filtering removes noise from daily fluctuations
    - **Confidence Intervals**: Statistical confidence bands around trend predictions
    - **Forecasting**: Projected weight trends based on current patterns
    - **Velocity Analysis**: Rate of weight change over time
    
    ### Chart Features
    - **Interactive Plots**: Zoom, pan, and hover for detailed information
    - **Multiple Views**: Raw data, smoothed trends, and forecasts
    - **Statistical Analysis**: Confidence intervals and trend analysis
    
    ### Settings
    - **Confidence Levels**: Adjust statistical confidence intervals (1σ, 95%)
    - **Forecast Period**: Set how far into the future to project trends (7-90 days)
    - **Aggregation Window**: Group measurements within time windows (0-24 hours)
    - **Residuals Analysis**: Configure histogram bins for error analysis
    """)
    
    # Body Composition Analysis
    st.markdown("""
    ## Body Composition Analysis
    
    Track and analyze your body composition metrics:
    
    ### Available Metrics
    - **Body Mass Index (BMI)**: Weight-to-height ratio indicator
    - **Fat-Free Mass Index (FFMI)**: Lean mass relative to height
    - **Body Fat Percentage**: Proportion of body weight that is fat
    - **Lean Body Mass**: Total weight minus fat mass
    
    ### Analysis Features
    - **Trend Visualization**: See how your composition changes over time
    - **Category Classifications**: Understand where you fall in standard ranges
    - **Correlation Analysis**: See relationships between different metrics
    - **Goal Tracking**: Monitor progress toward composition targets
    
    ### Interpretation Guidelines
    - **BMI Ranges**: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (≥30)
    - **FFMI Categories**: Below Average, Average, Above Average, Excellent, Exceptional
    - **Body Fat Categories**: Essential fat, Athletes, Fitness, Average, Above Average, Obese
    - **Note**: Categories are based on general guidelines; consult health professionals for personalized ranges
    """)
    
    # Settings and Configuration
    st.markdown("""
    ## Settings and Configuration
    
    ### User Profile
    - **Height Setting**: Required for BMI and FFMI calculations
    - **User Authentication**: Login/logout functionality for data security
    
    ### Analysis Settings
    - **Confidence Intervals**: Choose between 1σ and 95% confidence levels
    - **Forecasting**: Enable/disable future trend projections
    - **Forecast Duration**: Set how many days to project into the future
    - **Aggregation Window**: Group measurements within specified time windows
    
    ### Data Export
    - **Weights CSV**: Download weight data as CSV file
    - **LBM CSV**: Download lean body mass data as CSV file
    """)
    
    # Data Management
    st.markdown("""
    ## Data Management
    
    ### Importing Data
    1. Navigate to **Data Management** page
    2. Upload CSV files using the file uploaders
    3. Choose appropriate file type (Weights CSV, LBM CSV, or FITINDEX CSV)
    4. Ensure proper column headers (date, weight for weights; date, lbm for LBM)
    5. Files are automatically processed and imported
    
    ### Exporting Data
    1. Navigate to **Settings** page
    2. Scroll to **Data Export** section
    3. Click download buttons for available data types
    4. CSV files are automatically generated with current data
    
    ### Data Backup
    - **Regular Exports**: Download your data regularly for backup
    - **Multiple Formats**: Export in different formats for compatibility
    - **Cloud Storage**: Consider storing backups in cloud services
    
    ### File Requirements
    - **CSV Format**: Comma-separated values with headers
    - **Date Format**: YYYY-MM-DD or MM/DD/YYYY (auto-detected)
    - **Time Format**: HH:MM (24-hour format) or included in datetime
    - **Required Headers**: 
      - Weights: date, weight
      - LBM: date, lbm
      - FITINDEX: Time of Measurement, Weight(lb)
    """)
    
    # Understanding the Analytics
    st.markdown("""
    ## Understanding the Analytics
    
    ### Kalman Filter
    The application uses a Kalman filter for advanced trend analysis:
    - **Purpose**: Separates true trends from measurement noise
    - **Benefits**: More accurate trend identification and forecasting
    - **Parameters**: Automatically tuned for optimal performance
    
    ### Statistical Measures
    - **Confidence Intervals**: Range where true value likely falls
    - **Trend Velocity**: Rate of change over time
    - **Forecast Accuracy**: How reliable future predictions are
    
    ### Chart Interpretation
    - **Blue Dots**: Raw data points
    - **Red Line**: Smoothed trend (Kalman filter output)
    - **Shaded Area**: Confidence interval around trend
    - **Dashed Line**: Future forecast projection (if enabled)
    """)
    
    # Troubleshooting
    st.markdown("""
    ## Troubleshooting
    
    ### Common Issues
    
    **Data Not Appearing**
    - Check that you're logged in with the correct account
    - Verify data was saved successfully after entry
    - Refresh the page and try again
    
    **Charts Not Loading**
    - Ensure you have sufficient data points (minimum 3-5 entries)
    - Check your internet connection
    - Try clearing browser cache
    
    **Import Errors**
    - Verify CSV file format and headers
    - Check date format compatibility
    - Ensure no missing required fields
    
    **Performance Issues**
    - Large datasets may load slowly
    - Consider exporting old data to reduce file size
    - Close other browser tabs to free memory
    
    ### Getting Help
    - Check this User Guide for detailed instructions
    - Review the Disclaimer for important information
    - Contact support if issues persist
    """)
    
    # Best Practices
    st.markdown("""
    ## Best Practices
    
    ### Data Collection
    - **Consistent Timing**: Weigh yourself at the same time daily
    - **Same Conditions**: Use the same scale, clothing, and environment
    - **Regular Schedule**: Establish a routine for measurements
    - **Accurate Recording**: Double-check values before saving
    
    ### Analysis and Interpretation
    - **Focus on Trends**: Look at overall patterns, not daily fluctuations
    - **Consider Context**: Account for factors like hydration, meals, exercise
    - **Set Realistic Goals**: Aim for sustainable, healthy changes
    - **Professional Guidance**: Consult healthcare providers for major decisions
    
    ### Data Management
    - **Regular Backups**: Export data monthly
    - **Multiple Copies**: Store backups in different locations
    - **Version Control**: Keep track of data file versions
    - **Privacy**: Protect your personal health information
    
    ### Health and Safety
    - **Medical Consultation**: Seek professional advice for health decisions
    - **Realistic Expectations**: Understand that weight fluctuates naturally
    - **Holistic Approach**: Consider nutrition, exercise, and lifestyle factors
    - **Mental Health**: Maintain a healthy relationship with body metrics
    """)
    
    st.markdown("---")
    
    st.info("""
    **Note**: This application is designed for informational purposes only. Always consult with qualified healthcare professionals before making significant changes to your diet, exercise routine, or health management plan.
    """)

def show_disclaimer():
    """Display disclaimer and terms of use"""
    st.header("⚠️ Disclaimer and Terms of Use")
    
    st.markdown("""
    ### Important Notice
    
    This fitness tracking application is provided for informational purposes only. Please read the following disclaimers carefully before using this service.
    """)
    
    st.markdown("""
    #### Data Security and Reliability
    
    **No Data Guarantee**: We cannot guarantee that this application will be available at all times or that your data will be permanently preserved. Technical issues, server maintenance, or other unforeseen circumstances may result in temporary or permanent data loss.
    
    **Data Backup Recommendation**: We strongly recommend that you maintain your own backup copies of any important fitness data. Export your data regularly using the Data Management features to ensure you have local copies of your information.
    
    **Use at Your Own Risk**: You acknowledge and agree that you use this application at your own risk and that we are not responsible for any data loss, corruption, or unavailability.
    """)
    
    st.markdown("""
    #### Health and Medical Disclaimer
    
    **Not Medical Advice**: The information provided by this application is for general informational purposes only and is not intended as medical advice, diagnosis, or treatment. This application does not provide professional medical services.
    
    **Consult Healthcare Professionals**: Before making any significant changes to your diet, exercise routine, or weight management plan, you should consult with qualified healthcare professionals, including but not limited to:
    - Licensed physicians
    - Registered dietitians
    - Certified fitness professionals
    - Other appropriate healthcare providers
    
    **Individual Results May Vary**: Fitness and health outcomes vary significantly between individuals. The calculations, trends, and recommendations provided by this application are based on general formulas and may not be appropriate for your specific circumstances.
    
    **Not a Substitute for Professional Care**: This application is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    """)
    
    st.markdown("""
    #### Limitation of Liability
    
    **No Warranties**: This application is provided "as is" without any warranties, express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, or non-infringement.
    
    **Limitation of Damages**: In no event shall the developers, operators, or distributors of this application be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising out of or relating to your use of this application.
    """)
    
    st.markdown("""
    #### User Responsibilities
    
    **Accurate Data Entry**: You are responsible for ensuring the accuracy of all data you enter into this application. Inaccurate data will lead to inaccurate calculations and recommendations.
    
    **Regular Backups**: You are responsible for maintaining backup copies of your data and for ensuring the security of your account credentials.
    
    **Compliance with Terms**: By using this application, you agree to comply with all applicable terms of use and to use the application only for lawful purposes.
    """)
    
    st.markdown("""
    #### Contact Information
    
    If you have questions about this disclaimer or the application, please contact the development team through the appropriate channels.
    
    **Last Updated**: This disclaimer was last updated on the date of the current application version.
    """)
    
    st.warning("""
    **By continuing to use this application, you acknowledge that you have read, understood, and agree to be bound by this disclaimer and all applicable terms of use.**
    """)

if __name__ == "__main__":
    main()
