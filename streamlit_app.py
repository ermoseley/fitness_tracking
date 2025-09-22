#!/usr/bin/env python3
"""
Streamlit Fitness Tracking Application
Port of the Weight Tracker GUI to Streamlit
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

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our existing modules
from weight_tracker import (
    WeightEntry, load_entries, parse_datetime, parse_date,
    compute_time_aware_ema, fit_time_weighted_linear_regression,
    aggregate_entries, append_entry
)
from kalman import (
    WeightKalmanFilter, KalmanState, run_kalman_filter, 
    run_kalman_smoother, create_kalman_plot, create_bodyfat_plot_from_kalman,
    create_bmi_plot_from_kalman, create_ffmi_plot_from_kalman
)

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

# Page configuration
st.set_page_config(
    page_title="",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
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

def load_data_files():
    """Load data from CSV files"""
    data_dir = "data"
    weights_path = os.path.join(data_dir, "weights.csv")
    lbm_path = os.path.join(data_dir, "lbm.csv")
    height_path = os.path.join(data_dir, "height.txt")
    
    # Load weights data
    if os.path.exists(weights_path):
        st.session_state.weights_data = load_entries(weights_path)
        st.session_state.data_loaded = True
    else:
        st.session_state.weights_data = []
    
    # Load LBM data
    if os.path.exists(lbm_path):
        try:
            df = pd.read_csv(lbm_path)
            if 'date' in df.columns and 'lbm' in df.columns:
                st.session_state.lbm_data = df
            else:
                st.session_state.lbm_data = pd.DataFrame()
        except:
            st.session_state.lbm_data = pd.DataFrame()
    else:
        st.session_state.lbm_data = pd.DataFrame()
    
    # Load height
    if os.path.exists(height_path):
        try:
            with open(height_path, 'r') as f:
                st.session_state.height = float(f.read().strip())
        except:
            st.session_state.height = 67.0

def save_height(height_inches: float):
    """Save height to file"""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    height_path = os.path.join(data_dir, "height.txt")
    with open(height_path, 'w') as f:
        f.write(f"{height_inches:.3f}")
    st.session_state.height = height_inches

def main():
    # Load data on startup
    load_data_files()
    
    # Main header
    st.markdown('<h1 class="main-header"> Weight Tracker</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Dashboard", "📈 Weight Tracking", "📊 Body Composition", "⚙️ Settings", "📁 Data Management"]
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
    elif page == "📈 Weight Tracking":
        show_weight_tracking()
    elif page == "📊 Body Composition":
        show_body_composition()
    elif page == "⚙️ Settings":
        show_settings()
    elif page == "📁 Data Management":
        show_data_management()

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
                    "Current Weight",
                    f"{latest_kalman.weight:.1f} lbs",
                    f"±{ci_mult * latest_kalman.weight_var**0.5:.1f}"
                )
            
            with col2:
                velocity_per_week = 7 * latest_kalman.velocity
                st.metric(
                    "Weekly Rate",
                    f"{velocity_per_week:+.3f} lbs/week",
                    f"±{ci_mult * 7 * (latest_kalman.velocity_var**0.5):.3f}"
                )
            
            with col3:
                calorie_deficit = latest_kalman.velocity * 3500
                st.metric(
                    "Calorie Deficit",
                    f"{calorie_deficit:+.0f} cal/day",
                    "Estimated"
                )
            
            with col4:
                st.metric(
                    "Total Entries",
                    len(entries),
                    f"{(latest_entry.entry_datetime - first_entry.entry_datetime).days} days"
                )
            
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
            
            # Add dense Kalman estimate curve (historical portion)
            fig.add_trace(go.Scatter(
                x=dense_datetimes,
                y=dense_means,
                mode='lines',
                name='Kalman Estimate',
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
            
            fig.update_layout(
                title="Weight Trend with Kalman Filter Analysis" + (f" + {forecast_days}-Day Forecast" if st.session_state.enable_forecast else ""),
                xaxis_title="Date",
                yaxis_title="Weight (lbs)",
                hovermode='x unified',
                height=500
            )
            
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
                        f"{week_forecast:.2f} lbs",
                        f"±{ci_mult * week_std:.2f}"
                    )
                
                with col2:
                    month_forecast, month_std = kf.forecast(30.0)
                    st.metric(
                        "1-Month Forecast",
                        f"{month_forecast:.2f} lbs",
                        f"±{ci_mult * month_std:.2f}"
                    )
            
            # Recent entries table with Kalman estimates
            st.subheader("📋 Recent Entries")
            recent_entries = entries[-10:]  # Last 10 entries
            
            # Interpolate Kalman estimates to match entry dates using dense data
            from scipy.interpolate import interp1d
            kalman_interp = interp1d(
                [(d - dense_datetimes[0]).total_seconds() for d in dense_datetimes],
                dense_means,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            df_recent = pd.DataFrame([
                {
                    'Date': entry.entry_datetime.strftime('%Y-%m-%d %H:%M'),
                    'Weight (lbs)': f"{entry.weight:.1f}",
                    'Kalman Est.': f"{kalman_interp((entry.entry_datetime - dense_datetimes[0]).total_seconds()):.1f}"
                }
                for entry in recent_entries
            ])
            st.dataframe(df_recent, width='stretch')
            
        else:
            st.error("Failed to run Kalman filter analysis")
            return
            
    except Exception as e:
        st.error(f"Error running Kalman analysis: {e}")
        # Fallback to basic display
        st.metric("Latest Weight", f"{latest_entry.weight:.1f} lbs")
        st.metric("Total Entries", len(entries))
        return

def show_weight_tracking():
    """Weight tracking page with entry form and detailed analysis"""
    st.header("📈 Weight Tracking")
    
    # Add new weight entry
    st.subheader("➕ Add New Weight Entry")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Date and time input
        entry_date = st.date_input("Date", value=date.today())
        entry_time = st.time_input("Time", value=datetime.now().time())
        entry_datetime = datetime.combine(entry_date, entry_time)
    
    with col2:
        weight_value = st.number_input("Weight (lbs)", min_value=50.0, max_value=500.0, value=150.0, step=0.1)
        
        if st.button("Add Entry", type="primary"):
            if weight_value > 0:
                # Create weight entry
                entry = WeightEntry(entry_datetime, weight_value)
                
                # Save to CSV
                data_dir = "data"
                os.makedirs(data_dir, exist_ok=True)
                weights_path = os.path.join(data_dir, "weights.csv")
                append_entry(weights_path, entry)
                
                # Reload data
                load_data_files()
                st.success(f"Weight entry added: {weight_value} lbs on {entry_datetime.strftime('%Y-%m-%d %H:%M')}")
                st.rerun()
            else:
                st.error("Please enter a valid weight value")
    
    # Display current data
    if st.session_state.weights_data:
        st.subheader("📊 Weight Analysis (Kalman Filter)")
        
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
                        "Kalman Weight Estimate",
                        f"{latest_kalman.weight:.2f} lbs",
                        f"±{ci_mult * latest_kalman.weight_var**0.5:.2f}"
                    )
                
                with col2:
                    velocity_per_week = 7 * latest_kalman.velocity
                    st.metric(
                        "Weekly Rate",
                        f"{velocity_per_week:+.3f} lbs/week",
                        f"±{ci_mult * 7 * (latest_kalman.velocity_var**0.5):.3f}"
                    )
                
                with col3:
                    calorie_deficit = latest_kalman.velocity * 3500
                    st.metric(
                        "Calorie Deficit",
                        f"{calorie_deficit:+.0f} cal/day",
                        "Estimated"
                    )
                
                # Kalman plot
                st.subheader("🔬 Kalman Filter Analysis")
                
                # Create dense, smooth curves using the same approach as the original
                from kalman import compute_kalman_mean_std_spline
                dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(kalman_states, kalman_dates)
                
                # Create subplot with velocity
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Weight Estimate', 'Velocity (Rate of Change)'),
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3]
                )
                
                # Add original data points
                dates = [entry.entry_datetime for entry in st.session_state.weights_data]
                weights = [entry.weight for entry in st.session_state.weights_data]
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=weights,
                    mode='markers',
                    name='Weight Measurements',
                    marker=dict(size=8, color='blue', opacity=0.7)
                ), row=1, col=1)
                
                # Add dense Kalman estimate curve
                fig.add_trace(go.Scatter(
                    x=dense_datetimes,
                    y=dense_means,
                    mode='lines',
                    name='Kalman Estimate',
                    line=dict(color='red', width=2)
                ), row=1, col=1)
                
                # Add smooth confidence intervals
                ci_multiplier = get_confidence_multiplier(st.session_state.confidence_interval)
                upper_bound = [mean + ci_multiplier * std for mean, std in zip(dense_means, dense_stds)]
                lower_bound = [mean - ci_multiplier * std for mean, std in zip(dense_means, dense_stds)]
                
                fig.add_trace(go.Scatter(
                    x=dense_datetimes,
                    y=upper_bound,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=dense_datetimes,
                    y=lower_bound,
                    mode='lines',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(width=0),
                name=f'{st.session_state.confidence_interval} Confidence',
                hoverinfo='skip'
                ), row=1, col=1)
                
                # Add velocity plot with dense sampling
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
                dense_t = np.linspace(min_t, max_t, 5000)
                
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
                from datetime import timedelta
                dense_velocity_datetimes = [t0 + timedelta(days=float(td)) for td in dense_t]
                
                fig.add_trace(go.Scatter(
                    x=dense_velocity_datetimes,
                    y=dense_velocities,
                    mode='lines',
                    name='Velocity',
                    line=dict(color='green', width=2)
                ), row=2, col=1)
                
                # Add velocity confidence interval
                velocity_upper = [v + ci_multiplier * std for v, std in zip(dense_velocities, dense_velocity_stds)]
                velocity_lower = [v - ci_multiplier * std for v, std in zip(dense_velocities, dense_velocity_stds)]
                
                fig.add_trace(go.Scatter(
                    x=dense_velocity_datetimes,
                    y=velocity_upper,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=2, col=1)
                
                fig.add_trace(go.Scatter(
                    x=dense_velocity_datetimes,
                    y=velocity_lower,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(0,255,0,0.2)',
                    line=dict(width=0),
                    name=f'Velocity {st.session_state.confidence_interval} CI',
                    hoverinfo='skip'
                ), row=2, col=1)
                
                # Add zero line for velocity
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                
                fig.update_layout(
                    title="Kalman Filter Analysis: Weight and Velocity",
                    height=700,
                    hovermode='x unified'
                )
                
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Weight (lbs)", row=1, col=1)
                fig.update_yaxes(title_text="Velocity (lbs/day)", row=2, col=1)
                
                st.plotly_chart(fig, width='stretch')
                
                # Forecast section (only if enabled)
                if st.session_state.enable_forecast:
                    st.subheader("🔮 Forecast")
                    
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
                    
                    # Generate forecast data using configurable duration
                    forecast_days = int(st.session_state.forecast_days)
                    latest_date = ensure_py_datetime(kalman_dates[-1])
                    forecast_dates = [latest_date + timedelta(days=int(i)) for i in range(1, forecast_days + 1)]
                    forecast_weights = []
                    forecast_uncertainties = []
                    
                    for i in range(1, forecast_days + 1):
                        weight, std = kf.forecast(float(i))
                        forecast_weights.append(weight)
                        forecast_uncertainties.append(std)
                    
                    # Create forecast plot
                    forecast_fig = go.Figure()
                    
                    # Add historical data
                    forecast_fig.add_trace(go.Scatter(
                        x=dates,
                        y=weights,
                        mode='markers',
                        name='Historical Data',
                        marker=dict(size=6, color='blue', opacity=0.7)
                    ))
                    
                    # Add Kalman estimate (use dense, smooth curve)
                    forecast_fig.add_trace(go.Scatter(
                        x=dense_datetimes,
                        y=dense_means,
                        mode='lines',
                        name='Kalman Estimate',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Add forecast
                    forecast_fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_weights,
                        mode='lines',
                        name='Forecast',
                        line=dict(color='purple', width=2, dash='dash')
                    ))
                    
                    # Add forecast confidence interval
                    forecast_upper = [w + ci_multiplier * std for w, std in zip(forecast_weights, forecast_uncertainties)]
                    forecast_lower = [w - ci_multiplier * std for w, std in zip(forecast_weights, forecast_uncertainties)]
                    
                    forecast_fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_upper,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    forecast_fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_lower,
                        mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(128,0,128,0.2)',
                    line=dict(width=0),
                    name=f'Forecast {st.session_state.confidence_interval} CI',
                    hoverinfo='skip'
                    ))
                    
                    forecast_fig.update_layout(
                        title=f"{forecast_days}-Day Weight Forecast",
                        xaxis_title="Date",
                        yaxis_title="Weight (lbs)",
                        height=400
                    )
                    
                    st.plotly_chart(forecast_fig, width='stretch')
                    
                    # Forecast summary
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        week_forecast, week_std = kf.forecast(7.0)
                        st.metric(
                            "1-Week Forecast",
                            f"{week_forecast:.2f} lbs",
                            f"±{ci_mult * week_std:.2f}"
                        )
                    
                    with col2:
                        month_forecast, month_std = kf.forecast(30.0)
                        st.metric(
                            "1-Month Forecast",
                            f"{month_forecast:.2f} lbs",
                            f"±{ci_mult * month_std:.2f}"
                        )
        
        except Exception as e:
            st.error(f"Error running Kalman analysis: {e}")
    
    else:
        st.info("No weight data available. Please add some entries above.")

def show_body_composition():
    """Body composition analysis page"""
    st.header("📊 Body Composition Analysis")
    
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
            
            # BMI trend chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=kalman_dates,
                y=bmi_values,
                mode='lines+markers',
                name='BMI',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            # Add BMI category lines
            fig.add_hline(y=18.5, line_dash="dash", line_color="green", annotation_text="Underweight")
            fig.add_hline(y=25, line_dash="dash", line_color="yellow", annotation_text="Normal")
            fig.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Overweight")
            fig.add_hline(y=35, line_dash="dash", line_color="red", annotation_text="Obese")
            
            fig.update_layout(
                title="BMI Trend Over Time",
                xaxis_title="Date",
                yaxis_title="BMI",
                height=500
            )
            
            st.plotly_chart(fig, width='stretch')
    
    except Exception as e:
        st.error(f"Error calculating BMI: {e}")
    
    # Body fat analysis (if LBM data available)
    if not st.session_state.lbm_data.empty:
        st.subheader("🩸 Body Fat Analysis")
        
        try:
            # Process LBM data
            lbm_dates = pd.to_datetime(st.session_state.lbm_data['date'])
            lbm_values = st.session_state.lbm_data['lbm'].values
            
            # Calculate body fat percentage
            # Need to interpolate LBM to match weight dates
            weight_dates = [entry.entry_datetime for entry in st.session_state.weights_data]
            weights = [entry.weight for entry in st.session_state.weights_data]
            
            # Interpolate LBM to weight measurement dates
            from scipy.interpolate import interp1d
            lbm_interp = interp1d(
                (lbm_dates - lbm_dates[0]).dt.total_seconds(),
                lbm_values,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            bf_percentages = []
            lbm_interpolated = []
            for i, (w_date, weight) in enumerate(zip(weight_dates, weights)):
                if i == 0:
                    continue
                time_diff = (w_date - weight_dates[0]).total_seconds()
                lbm = lbm_interp(time_diff)
                lbm_interpolated.append(lbm)
                bf_percent = ((weight - lbm) / weight) * 100
                bf_percentages.append(bf_percent)
            
            if bf_percentages:
                # Display current body fat
                current_bf = bf_percentages[-1]
                current_lbm = lbm_interpolated[-1]
                bf_category = get_bf_category(current_bf, 'male')  # Assuming male for now
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Body Fat %", f"{current_bf:.1f}%")
                
                with col2:
                    st.metric("Body Fat Category", bf_category)
                
                with col3:
                    st.metric("Current LBM", f"{current_lbm:.1f} lbs")
                
                # Body fat trend chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=weight_dates[1:],  # Skip first point since we can't calculate BF
                    y=bf_percentages,
                    mode='lines+markers',
                    name='Body Fat %',
                    line=dict(color='purple', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title="Body Fat Percentage Trend",
                    xaxis_title="Date",
                    yaxis_title="Body Fat %",
                    height=500
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # FFMI Analysis
                st.subheader("💪 FFMI Analysis")
                
                # Calculate FFMI (Fat-Free Mass Index)
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
                
                # FFMI trend chart
                ffmi_fig = go.Figure()
                
                ffmi_fig.add_trace(go.Scatter(
                    x=weight_dates[1:],
                    y=ffmi_values,
                    mode='lines+markers',
                    name='FFMI',
                    line=dict(color='green', width=2),
                    marker=dict(size=6)
                ))
                
                # Add FFMI reference lines
                ffmi_fig.add_hline(y=16, line_dash="dash", line_color="red", annotation_text="Below Average")
                ffmi_fig.add_hline(y=18, line_dash="dash", line_color="orange", annotation_text="Average")
                ffmi_fig.add_hline(y=20, line_dash="dash", line_color="yellow", annotation_text="Above Average")
                ffmi_fig.add_hline(y=22, line_dash="dash", line_color="green", annotation_text="Excellent")
                
                ffmi_fig.update_layout(
                    title="Fat-Free Mass Index (FFMI) Trend",
                    xaxis_title="Date",
                    yaxis_title="FFMI",
                    height=500
                )
                
                st.plotly_chart(ffmi_fig, width='stretch')
        
        except Exception as e:
            st.error(f"Error calculating body fat: {e}")
    
    else:
        st.info("No LBM data available. Upload a CSV file with 'date' and 'lbm' columns to see body fat analysis.")

def show_settings():
    """Settings and configuration page"""
    st.header("⚙️ Settings")
    
    st.subheader("🔧 Kalman Filter Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        process_noise_weight = st.slider(
            "Process Noise (Weight)",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            help="Higher values allow more weight variation"
        )
        
        process_noise_velocity = st.slider(
            "Process Noise (Velocity)",
            min_value=0.0001,
            max_value=0.01,
            value=0.001,
            step=0.0001,
            help="Higher values allow more velocity variation"
        )
    
    with col2:
        measurement_noise = st.slider(
            "Measurement Noise",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Expected measurement uncertainty"
        )
        
        confidence_interval = st.selectbox(
            "Confidence Interval",
            ["1σ", "95%"],
            index=0 if st.session_state.confidence_interval == "1σ" else 1,
            help="Confidence level for uncertainty bands"
        )
        
        # Update session state immediately when changed
        if confidence_interval != st.session_state.confidence_interval:
            st.session_state.confidence_interval = confidence_interval
            st.rerun()
    
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
        
        # Update session state immediately when changed
        if enable_forecast != st.session_state.enable_forecast:
            st.session_state.enable_forecast = enable_forecast
            st.rerun()
        
        if forecast_days != st.session_state.forecast_days:
            st.session_state.forecast_days = forecast_days
            st.rerun()
    
    with col2:
        st.info("**Note**: This app uses Kalman filtering as the primary analysis method, which automatically provides optimal smoothing and trend estimation.")
    
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
    
    col1, col2 = st.columns(2)
    
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
                    # Save to data directory
                    data_dir = "data"
                    os.makedirs(data_dir, exist_ok=True)
                    weights_path = os.path.join(data_dir, "weights.csv")
                    df.to_csv(weights_path, index=False)
                    
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
                    
                    # Save to data directory
                    data_dir = "data"
                    os.makedirs(data_dir, exist_ok=True)
                    lbm_path = os.path.join(data_dir, "lbm.csv")
                    df.to_csv(lbm_path, index=False)
                    
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
                                # Save the fixed file
                                data_dir = "data"
                                os.makedirs(data_dir, exist_ok=True)
                                lbm_path = os.path.join(data_dir, "lbm.csv")
                                df_fixed.to_csv(lbm_path, index=False)
                                
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
    
    # Data preview
    if st.session_state.weights_data:
        st.subheader("📋 Weight Data Preview")
        recent_weights = st.session_state.weights_data[-10:]
        weights_df = pd.DataFrame([
            {
                'Date': entry.entry_datetime.strftime('%Y-%m-%d %H:%M'),
                'Weight (lbs)': f"{entry.weight:.1f}"
            }
            for entry in recent_weights
        ])
        st.dataframe(weights_df, width='stretch')
    
    if not st.session_state.lbm_data.empty:
        st.subheader("📋 LBM Data Preview")
        st.dataframe(st.session_state.lbm_data.tail(10), width='stretch')

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

if __name__ == "__main__":
    main()
