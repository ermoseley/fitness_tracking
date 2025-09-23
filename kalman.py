#!/usr/bin/env python3

import os
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.interpolate import CubicSpline
from scipy import stats
import matplotlib.pyplot as plt
import csv


@dataclass
class KalmanState:
    """Kalman filter state for weight tracking"""
    weight: float  # Current weight estimate
    velocity: float  # Rate of weight change (lbs/day)
    weight_var: float  # Variance of weight estimate
    velocity_var: float  # Variance of velocity estimate
    weight_velocity_cov: float  # Covariance between weight and velocity


class WeightKalmanFilter:
    """
    Kalman filter for weight tracking with position-velocity model.
    
    State vector: [weight, velocity]
    Measurement: weight only
    """
    
    def __init__(self, 
                 initial_weight: float,
                 initial_velocity: float = 0.0,
                 process_noise_weight: float = 0.01,
                 process_noise_velocity: float = 0.001,
                 process_noise_offdiag: float = 0.0,
                 measurement_noise: float = 0.5,
                 initial_weight_var: Optional[float] = None,
                 initial_velocity_var: float = 0.25):
        """
        Initialize Kalman filter
        
        Args:
            initial_weight: Starting weight estimate
            initial_velocity: Starting velocity estimate (lbs/day)
            process_noise_weight: Process noise for weight (variance per day)
            process_noise_velocity: Process noise for velocity (variance per day)
            measurement_noise: Measurement noise variance
        """
        self.measurement_noise = float(measurement_noise)
        
        # Process noise matrix per day (2x2), allow off-diagonal coupling
        self.Q_day = np.array([
            [process_noise_weight, process_noise_offdiag],
            [process_noise_offdiag, process_noise_velocity]
        ], dtype=float)
        
        # State vector and covariance matrix (matrix form)
        self.x = np.array([float(initial_weight), float(initial_velocity)], dtype=float)
        if initial_weight_var is None:
            initial_weight_var = float(self.measurement_noise)
        self.P = np.array([
            [float(initial_weight_var), 0.0],
            [0.0, float(initial_velocity_var)]
        ], dtype=float)
        
        # Measurement model and noise
        self.H = np.array([[1.0, 0.0]], dtype=float)
        self.R = np.array([[self.measurement_noise]], dtype=float)
        # Adaptive measurement noise (EWMA on innovation variance)
        self.R_adapt = float(self.measurement_noise)
        self.R_alpha = 0.05
        self.adaptive_measurement_noise = True
        
        # Expose a convenience dataclass mirror for external access
        self.state = KalmanState(
            weight=float(self.x[0]),
            velocity=float(self.x[1]),
            weight_var=float(self.P[0, 0]),
            velocity_var=float(self.P[1, 1]),
            weight_velocity_cov=float(self.P[0, 1])
        )
    
    def predict(self, dt_days: float) -> None:
        """
        Predict state forward by dt_days
        
        Args:
            dt_days: Time step in days
        """
        # State transition matrix
        F = np.array([
            [1.0, dt_days],
            [0.0, 1.0]
        ], dtype=float)
        
        # Predict state and covariance (matrix form)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q_day * float(dt_days)
        
        # Mirror to convenience dataclass
        self.state.weight = float(self.x[0])
        self.state.velocity = float(self.x[1])
        self.state.weight_var = float(self.P[0, 0])
        self.state.velocity_var = float(self.P[1, 1])
        self.state.weight_velocity_cov = float(self.P[0, 1])
    
    def update(self, measurement: float) -> None:
        """
        Update state with new measurement
        
        Args:
            measurement: New weight measurement
        """
        # Innovation
        z = np.array([[float(measurement)]], dtype=float)
        y = z - self.H @ self.x.reshape(-1, 1)
        # Adapt measurement noise from innovation statistics (pre-update)
        if self.adaptive_measurement_noise:
            # S_est ≈ y^2, so R_est ≈ S_est - HPH^T
            S_est = float(y @ y.T)
            HPH = float(self.H @ self.P @ self.H.T)
            R_est = max(1e-8, S_est - HPH)
            self.R_adapt = (1.0 - self.R_alpha) * self.R_adapt + self.R_alpha * R_est
            self.R = np.array([[self.R_adapt]], dtype=float)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = (self.x.reshape(-1, 1) + K @ y).flatten()
        I = np.eye(2)
        # Joseph form for numerical stability
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        
        # Mirror to convenience dataclass
        self.state.weight = float(self.x[0])
        self.state.velocity = float(self.x[1])
        self.state.weight_var = float(self.P[0, 0])
        self.state.velocity_var = float(self.P[1, 1])
        self.state.weight_velocity_cov = float(self.P[0, 1])
    
    def forecast(self, days_ahead: float) -> Tuple[float, float]:
        """
        Forecast weight and uncertainty for a future date
        
        Args:
            days_ahead: Days in the future to forecast
            
        Returns:
            (forecasted_weight, forecasted_std)
        """
        # Work on a copy so we do not mutate the original state object
        # Copy state and covariance for simulation
        x0 = self.x.copy()
        P0 = self.P.copy()
        # Predict forward
        self.predict(days_ahead)
        forecast_weight = float(self.x[0])
        forecast_std = float(np.sqrt(max(self.P[0, 0], 0.0)))
        # Restore
        self.x = x0
        self.P = P0
        # Keep mirror consistent
        self.state.weight = float(self.x[0])
        self.state.velocity = float(self.x[1])
        self.state.weight_var = float(self.P[0, 0])
        self.state.velocity_var = float(self.P[1, 1])
        self.state.weight_velocity_cov = float(self.P[0, 1])
        return forecast_weight, forecast_std


def run_kalman_filter(entries, 
                     initial_weight: Optional[float] = None,
                     initial_velocity: float = 0.0) -> Tuple[List[KalmanState], List[date]]:
    """
    Run Kalman filter on weight entries
    
    Args:
        entries: List of WeightEntry objects
        initial_weight: Starting weight (defaults to first measurement)
        initial_velocity: Starting velocity estimate
        
    Returns:
        (kalman_states, dates)
    """
    if not entries:
        return [], []
    
    # Initialize filter
    if initial_weight is None:
        initial_weight = entries[0].weight
    
    kf = WeightKalmanFilter(
        initial_weight=initial_weight,
        initial_velocity=initial_velocity
    )
    
    states = []
    dates = []
    
    prev_datetime = None
    
    for entry in entries:
        if prev_datetime is not None:
            # Predict forward to current datetime
            dt_days = (entry.entry_datetime - prev_datetime).total_seconds() / 86400.0
            if dt_days > 0:
                kf.predict(dt_days)
        
        # Update with measurement
        kf.update(entry.weight)
        
        # Store state
        states.append(KalmanState(
            weight=kf.state.weight,
            velocity=kf.state.velocity,
            weight_var=kf.state.weight_var,
            velocity_var=kf.state.velocity_var,
            weight_velocity_cov=kf.state.weight_velocity_cov
        ))
        dates.append(entry.entry_datetime)
        
        prev_datetime = entry.entry_datetime
    
    return states, dates


def run_kalman_smoother(entries,
                        initial_weight: Optional[float] = None,
                        initial_velocity: float = 0.0) -> Tuple[List[KalmanState], List[date]]:
    """
    Run Rauch-Tung-Striebel smoother on weight entries.

    This performs a full forward pass (Kalman filter) while storing the
    necessary transition and prediction terms, followed by a backward
    smoothing pass to compute optimal retrospective estimates that use
    both past and future information.

    Returns:
        (smoothed_states, dates)
    """
    if not entries:
        return [], []

    # Initialize filter
    if initial_weight is None:
        initial_weight = entries[0].weight

    kf = WeightKalmanFilter(
        initial_weight=initial_weight,
        initial_velocity=initial_velocity
    )

    filtered_x_list: List[np.ndarray] = []
    filtered_P_list: List[np.ndarray] = []
    dates: List[date] = []

    # Store per-step transitions/predictions for RTS
    F_list: List[np.ndarray] = []  # state transition from k->k+1
    x_pred_list: List[np.ndarray] = []  # x_{k+1|k}
    P_pred_list: List[np.ndarray] = []  # P_{k+1|k}

    prev_datetime: Optional[datetime] = None

    for idx, entry in enumerate(entries):
        if idx == 0:
            # First update at t0
            kf.update(entry.weight)
            filtered_x_list.append(kf.x.copy())
            filtered_P_list.append(kf.P.copy())
            dates.append(entry.entry_datetime)
            prev_datetime = entry.entry_datetime
            continue

        # Time delta in days (allow zero for repeated same-time entries)
        dt_days = max(0, (entry.entry_datetime - prev_datetime).total_seconds() / 86400.0 if prev_datetime is not None else 0)
        # Transition used in predict step
        F = np.array([[1.0, float(dt_days)], [0.0, 1.0]], dtype=float)

        # Predict to current timestamp and record predicted terms
        kf.predict(float(dt_days))
        F_list.append(F)
        x_pred_list.append(kf.x.copy())
        P_pred_list.append(kf.P.copy())

        # Measurement update
        kf.update(entry.weight)
        filtered_x_list.append(kf.x.copy())
        filtered_P_list.append(kf.P.copy())
        dates.append(entry.entry_datetime)
        prev_datetime = entry.entry_datetime

    n = len(filtered_x_list)
    if n == 0:
        return [], []

    # Initialize smoothed estimates with filtered terminal state
    x_smooth: List[np.ndarray] = [None] * n  # type: ignore
    P_smooth: List[np.ndarray] = [None] * n  # type: ignore
    x_smooth[-1] = filtered_x_list[-1].copy()
    P_smooth[-1] = filtered_P_list[-1].copy()

    # Backward RTS smoothing
    for k in range(n - 2, -1, -1):
        Pk = filtered_P_list[k]
        Fk = F_list[k]
        P_pred = P_pred_list[k]
        x_pred = x_pred_list[k]

        # Smoother gain J_k = P_k F_k^T (P_{k+1|k})^{-1}
        try:
            P_pred_inv = np.linalg.inv(P_pred)
        except np.linalg.LinAlgError:
            P_pred_inv = np.linalg.pinv(P_pred)
        Jk = Pk @ Fk.T @ P_pred_inv

        # x_k|N = x_k|k + J_k (x_{k+1|N} - x_{k+1|k})
        delta_x = x_smooth[k + 1] - x_pred
        x_smooth[k] = filtered_x_list[k] + Jk @ delta_x

        # P_k|N = P_k|k + J_k (P_{k+1|N} - P_{k+1|k}) J_k^T
        P_smooth[k] = Pk + Jk @ (P_smooth[k + 1] - P_pred) @ Jk.T

    # Convert to KalmanState list
    smoothed_states: List[KalmanState] = []
    for xs, Ps in zip(x_smooth, P_smooth):
        w = float(xs[0])
        v = float(xs[1])
        wv = float(Ps[0, 1])
        smoothed_states.append(KalmanState(
            weight=w,
            velocity=v,
            weight_var=float(max(Ps[0, 0], 0.0)),
            velocity_var=float(max(Ps[1, 1], 0.0)),
            weight_velocity_cov=wv,
        ))

    return smoothed_states, dates

def interpolate_kalman_states(states, 
                            dates,
                            target_dates) -> Tuple[List[float], List[float], List[float]]:
    """
    Interpolate Kalman filter states to target dates using cubic splines
    
    Args:
        states: List of KalmanState objects
        dates: List of dates corresponding to states
        target_dates: List of target dates for interpolation
        
    Returns:
        (interpolated_weights, interpolated_velocities, interpolated_weight_stds)
    """
    if len(states) < 2:
        # Not enough data for interpolation
        if states:
            weight = states[0].weight
            velocity = states[0].velocity
            std = np.sqrt(states[0].weight_var)
            return [weight] * len(target_dates), [velocity] * len(target_dates), [std] * len(target_dates)
        else:
            return [], [], []
    
    # Convert datetimes to days from first datetime for numerical interpolation
    t0 = dates[0]
    t_original = [(d - t0).total_seconds() / 86400.0 for d in dates]
    t_target = [(d - t0).total_seconds() / 86400.0 for d in target_dates]
    
    # Ensure strictly increasing time values
    if len(set(t_original)) != len(t_original):
        print(f"Warning: Duplicate dates detected, using unique time points")
        # Create unique time-value pairs
        unique_data = {}
        for i, t in enumerate(t_original):
            if t not in unique_data:
                unique_data[t] = {
                    'weights': [], 'velocities': [], 'stds': []
                }
            unique_data[t]['weights'].append(states[i].weight)
            unique_data[t]['velocities'].append(states[i].velocity)
            unique_data[t]['stds'].append(np.sqrt(states[i].weight_var))
        
        # Average values for duplicate times
        t_original = sorted(unique_data.keys())
        weights = [np.mean(unique_data[t]['weights']) for t in t_original]
        velocities = [np.mean(unique_data[t]['velocities']) for t in t_original]
        weight_stds = [np.mean(unique_data[t]['stds']) for t in t_original]
    else:
        # Extract values directly
        weights = [s.weight for s in states]
        velocities = [s.velocity for s in states]
        weight_stds = [np.sqrt(s.weight_var) for s in states]
    
    # Create splines: prefer PCHIP for stds to keep non-negativity/shape
    try:
        weight_spline = CubicSpline(t_original, weights, bc_type='natural')
        velocity_spline = CubicSpline(t_original, velocities, bc_type='natural')
        try:
            from scipy.interpolate import PchipInterpolator  # type: ignore
            std_spline_fn = PchipInterpolator(t_original, weight_stds)
            interpolated_stds = std_spline_fn(t_target)
        except Exception:
            std_spline = CubicSpline(t_original, weight_stds, bc_type='natural')
            interpolated_stds = std_spline(t_target)

        # Interpolate
        interpolated_weights = weight_spline(t_target)
        interpolated_velocities = velocity_spline(t_target)

        # Ensure numpy arrays
        if not isinstance(interpolated_weights, np.ndarray):
            interpolated_weights = np.array(interpolated_weights)
        if not isinstance(interpolated_velocities, np.ndarray):
            interpolated_velocities = np.array(interpolated_velocities)
        if not isinstance(interpolated_stds, np.ndarray):
            interpolated_stds = np.array(interpolated_stds)
    except Exception as e:
        print(f"Warning: Spline failed ({e}), falling back to linear interpolation")
        interpolated_weights = np.interp(t_target, t_original, weights)
        interpolated_velocities = np.interp(t_target, t_original, velocities)
        interpolated_stds = np.interp(t_target, t_original, weight_stds)
    
    return interpolated_weights.tolist(), interpolated_velocities.tolist(), interpolated_stds.tolist()


def compute_kalman_mean_std_spline(states, dates, n_points: int = 10000) -> Tuple[List[datetime], List[float], List[float]]:
    """
    Mirror the EMA spline approach from weight_tracker.py for Kalman outputs.
    Build a dense cubic spline for the filtered mean and for the std, using
    datetimes and a dense time grid for smooth plotting.

    Args:
        states: List of Kalman states
        dates: List of datetime objects
        n_points: Number of points for dense sampling (default: 10000)

    Returns: (dense_datetimes, dense_mean_values, dense_std_values)
    """
    if not states:
        return [], [], []

    # Use datetimes directly for smooth plotting and collapse duplicate datetimes by averaging
    entry_datetimes = [d for d in dates]

    from collections import defaultdict
    dt_to_means = defaultdict(list)
    dt_to_stds = defaultdict(list)
    for dt, st in zip(entry_datetimes, states):
        dt_to_means[dt].append(float(st.weight))
        dt_to_stds[dt].append(float(np.sqrt(max(st.weight_var, 0.0))))

    unique_datetimes = sorted(dt_to_means.keys())
    mean_vals_unique = [float(np.mean(dt_to_means[dt])) for dt in unique_datetimes]
    std_vals_unique = [float(np.mean(dt_to_stds[dt])) for dt in unique_datetimes]

    # If only one unique point, return constant series
    if len(unique_datetimes) == 1:
        return [unique_datetimes[0]], [mean_vals_unique[0]], [std_vals_unique[0]]

    # Time axis in fractional days
    t0 = unique_datetimes[0]
    t_entry_days = np.array([(dt - t0).total_seconds() / 86400.0 for dt in unique_datetimes], dtype=float)
    mean_vals = np.array(mean_vals_unique, dtype=float)
    std_vals = np.array(std_vals_unique, dtype=float)

    # Dense grid as in EMA (smooth curve)
    min_t = float(np.min(t_entry_days))
    max_t = float(np.max(t_entry_days))
    dense_t = np.linspace(min_t, max_t, n_points)

    # Spline the mean and std (natural boundary), with robust fallbacks
    try:
        from scipy.interpolate import CubicSpline  # type: ignore
        mean_spline = CubicSpline(t_entry_days, mean_vals, bc_type="natural")
        std_spline = CubicSpline(t_entry_days, std_vals, bc_type="natural")
        dense_mean = mean_spline(dense_t)
        dense_std = std_spline(dense_t)
    except Exception:
        try:
            from scipy.interpolate import PchipInterpolator  # type: ignore
            mean_pchip = PchipInterpolator(t_entry_days, mean_vals)
            std_pchip = PchipInterpolator(t_entry_days, std_vals)
            dense_mean = mean_pchip(dense_t)
            dense_std = std_pchip(dense_t)
        except Exception:
            dense_mean = np.interp(dense_t, t_entry_days, mean_vals)
            dense_std = np.interp(dense_t, t_entry_days, std_vals)

    from datetime import timedelta
    dense_datetimes = [t0 + timedelta(days=float(td)) for td in dense_t]

    return dense_datetimes, [float(v) for v in dense_mean], [float(max(float(s), 0.0)) for s in dense_std]


def _load_lbm_csv(path: str) -> List[Tuple[date, float]]:
    points: List[Tuple[date, float]] = []
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    d_str = str(row[0]).strip()
                    v_str = str(row[1]).strip()
                    d = datetime.fromisoformat(d_str).date()
                    v = float(v_str)
                except Exception:
                    # skip header or bad rows
                    continue
                points.append((d, v))
    except Exception:
        return []
    points.sort(key=lambda p: p[0])
    return points


def _evaluate_lbm_series(target_datetimes: List[datetime], lbm_points: List[Tuple[date, float]]) -> List[float]:
    if not lbm_points:
        return [0.0 for _ in target_datetimes]
    # Build lists of datetimes and values
    pts_dt = [datetime.combine(d, datetime.min.time()) for d, _ in lbm_points]
    pts_val = [float(v) for _, v in lbm_points]

    # For each target time, piecewise linear interpolate; constant after last point
    out: List[float] = []
    for t in target_datetimes:
        if t <= pts_dt[0]:
            out.append(pts_val[0])
            continue
        if t >= pts_dt[-1]:
            out.append(pts_val[-1])
            continue
        # find the interval
        idx = 1
        while idx < len(pts_dt) and not (pts_dt[idx-1] <= t <= pts_dt[idx]):
            idx += 1
        if idx >= len(pts_dt):
            out.append(pts_val[-1])
            continue
        t0, t1 = pts_dt[idx-1], pts_dt[idx]
        v0, v1 = pts_val[idx-1], pts_val[idx]
        # linear interpolation
        span = (t1 - t0).total_seconds()
        alpha = 0.0 if span <= 0 else (t - t0).total_seconds() / span
        out.append((1.0 - alpha) * v0 + alpha * v1)
    return out

def create_kalman_plot(entries, 
                      states, 
                      dates,
                      output_path: str,
                      no_display: bool = False,
                      label: str = "Kalman Filter Estimate",
                      start_date: Optional[date] = None,
                      end_date: Optional[date] = None,
                      ci_multiplier: float = 1.96,
                      enable_forecast: bool = True) -> None:
    """
    Create Kalman filter plot with raw data, filtered state, and confidence bands
    
    Args:
        entries: List of WeightEntry objects
        states: List of KalmanState objects
        dates: List of dates corresponding to states
        output_path: Path to save the plot
    """
    from datetime import datetime as dt, timedelta
    
    if not entries or not states:
        return
    
    # Filter entries by date range if specified
    filtered_entries = entries
    if start_date or end_date:
        if start_date:
            # Convert start_date to datetime at beginning of day
            start_datetime = dt.combine(start_date, dt.min.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime >= start_datetime]
        if end_date:
            # Convert end_date to datetime at end of day
            end_datetime = dt.combine(end_date, dt.max.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime <= end_datetime]
        
        if not filtered_entries:
            print(f"Warning: No data in specified date range {start_date} to {end_date}")
            return
    
    # Build smooth dense curves exactly like the EMA spline approach (date-based x-axis)
    dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(states, dates)
    
    # Filter dense curves by date range if specified
    if start_date or end_date:
        filtered_dense_datetimes = []
        filtered_dense_means = []
        filtered_dense_stds = []
        for dt, mean, std in zip(dense_datetimes, dense_means, dense_stds):
            dt_date = dt.date()
            if start_date and dt_date < start_date:
                continue
            if end_date and dt_date > end_date:
                continue
            filtered_dense_datetimes.append(dt)
            filtered_dense_means.append(mean)
            filtered_dense_stds.append(std)
        dense_datetimes = filtered_dense_datetimes
        dense_means = filtered_dense_means
        dense_stds = filtered_dense_stds
    
    entry_datetimes = [e.entry_datetime for e in entries]
    
    # Create plot with high-quality settings
    plt.figure(figsize=(12, 8), dpi=100)
    
    # Set matplotlib parameters for smooth curves (non-intrusive)
    plt.rcParams['savefig.dpi'] = 150
    
    # Plot confidence bands
    upper_band = [m + ci_multiplier * s for m, s in zip(dense_means, dense_stds)]
    lower_band = [m - ci_multiplier * s for m, s in zip(dense_means, dense_stds)]
    
    # Fill confidence bands with anti-aliasing for smooth curves
    ci_label = f"{ci_multiplier:.1f}σ Confidence Interval" if ci_multiplier == 1.0 else f"{ci_multiplier/1.96*95:.0f}% Confidence Interval"
    plt.fill_between(dense_datetimes, lower_band, upper_band, 
                     alpha=0.3, color='gray', label=ci_label,
                     antialiased=True, linewidth=0)
    
    # Plot filtered/smoothed state mean with anti-aliasing for smooth curves
    plt.plot(dense_datetimes, dense_means, '-', color='#ff7f0e', 
             linewidth=2, label=label, 
             antialiased=True, solid_capstyle='round')
    
    # Plot raw data (filter to selected date range for display)
    plotted_entries = filtered_entries if (start_date or end_date) else entries
    entry_datetimes = [e.entry_datetime for e in plotted_entries]
    raw_weights = [e.weight for e in plotted_entries]
    plt.scatter(entry_datetimes, raw_weights, s=50, color='blue', 
                alpha=0.7, label='Raw Measurements', zorder=5)
    
    # Add stats box
    latest_state = states[-1]
    latest_date = dates[-1]
    # Use the spline-smoothed current mean/std so the panel matches the curve
    current_mean = float(dense_means[-1]) if dense_means else float(latest_state.weight)
    current_std = float(max(dense_stds[-1], 0.0)) if dense_stds else float(np.sqrt(max(latest_state.weight_var, 0.0)))
    
    # Calculate forecasts
    kf = WeightKalmanFilter(
        initial_weight=latest_state.weight,
        initial_velocity=latest_state.velocity,
        initial_weight_var=latest_state.weight_var,
        initial_velocity_var=latest_state.velocity_var,
    )
    # Set full covariance including off-diagonal from latest state
    kf.x = np.array([latest_state.weight, latest_state.velocity], dtype=float)
    kf.P = np.array([
        [latest_state.weight_var, latest_state.weight_velocity_cov],
        [latest_state.weight_velocity_cov, latest_state.velocity_var],
    ], dtype=float)
    
    # Forecast 1 week and 1 month ahead
    week_forecast, week_std = kf.forecast(7.0)
    month_forecast, month_std = kf.forecast(30.0)
    
    # Calculate velocity uncertainty
    velocity_std = np.sqrt(max(latest_state.velocity_var, 0.0))
    velocity_ci = ci_multiplier * velocity_std
    velocity_per_week = 7 * latest_state.velocity
    velocity_ci_per_week = 7 * velocity_ci
    
    # Create stats text (mean/std from spline; slope from latest Kalman state)
    stats_text = f"""Current Estimate ({latest_date})
Weight: {current_mean:.2f} ± {ci_multiplier * current_std:.2f}
Rate: {velocity_per_week:+.3f} ± {velocity_ci_per_week:.3f} lbs/week

Forecasts:
1 week: {week_forecast:.2f} ± {ci_multiplier * week_std:.2f}
1 month: {month_forecast:.2f} ± {ci_multiplier * month_std:.2f}"""
    
    # Add forecasting fan if enabled
    if enable_forecast:  # Show forecast even when filtering by date
        # Create forecast dates (1 month = 30 days) - start from day 0 to eliminate gap
        forecast_days = 30
        last_date = dense_datetimes[-1]
        forecast_dates = [last_date + timedelta(days=i) for i in range(0, forecast_days + 1)]
        
        # Calculate forecast weights and uncertainty
        forecast_weights = []
        forecast_upper = []
        forecast_lower = []
        
        for i in range(0, forecast_days + 1):
            w_forecast, w_forecast_std = kf.forecast(float(i))
            forecast_weights.append(w_forecast)
            forecast_upper.append(w_forecast + ci_multiplier * w_forecast_std)
            forecast_lower.append(w_forecast - ci_multiplier * w_forecast_std)
        
        # Plot forecast fan
        ci_label = "95%" if abs(ci_multiplier - 1.96) < 0.01 else "1σ"
        plt.fill_between(forecast_dates, forecast_lower, forecast_upper, 
                        alpha=0.2, color='gray')
        
        # Plot forecast mean line - use same color as main plot line
        plt.plot(forecast_dates, forecast_weights, '--', color='#ff7f0e', linewidth=1.5, 
                alpha=0.8, label='Weight Forecast')

    # Add stats box
    ax = plt.gca()
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
            ha='left', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', 
                     alpha=0.9, edgecolor='#cccccc'), zorder=10)
    
    plt.title('Weight Trend - Kalman Filter')
    plt.xlabel('Date')
    plt.ylabel('Weight (lbs)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set x-axis limits based on filtered dense curve for a natural view
    dense_min = min(dense_datetimes) if dense_datetimes else (min(entry_datetimes) if entry_datetimes else None)
    dense_max = max(dense_datetimes) if dense_datetimes else (max(entry_datetimes) if entry_datetimes else None)
    if start_date or end_date:
        left = dt.combine(start_date, dt.min.time()) if start_date else dense_min
        right = dt.combine(end_date, dt.max.time()) if end_date else dense_max
    else:
        left, right = dense_min, dense_max
    if left is not None and right is not None and left <= right:
        span = right - left
        pad = max(span * 0.02, timedelta(days=1))
        left_pad = left if start_date else (left - pad)
        # Always extend 30 days into the future, regardless of date filtering
        right_pad = right + timedelta(days=30) if end_date else (right + timedelta(days=30))
        plt.xlim(left=left_pad, right=right_pad)
    
    # After setting x-limits, set Y-limits based on visible series (bands, mean, points)
    try:
        y_values: List[float] = []
        if lower_band and upper_band:
            y_values.extend([float(v) for v in lower_band])
            y_values.extend([float(v) for v in upper_band])
        if dense_means:
            y_values.extend([float(v) for v in dense_means])
        if raw_weights:
            y_values.extend([float(v) for v in raw_weights])
        # Include forecast values in y-axis range calculation
        if enable_forecast and 'forecast_weights' in locals():
            y_values.extend([float(v) for v in forecast_weights])
            y_values.extend([float(v) for v in forecast_upper])
            y_values.extend([float(v) for v in forecast_lower])
        if y_values:
            y_min = float(np.nanmin(np.array(y_values, dtype=float)))
            y_max = float(np.nanmax(np.array(y_values, dtype=float)))
            if not np.isfinite(y_min) or not np.isfinite(y_max):
                raise ValueError("Non-finite y-range")
            if y_min == y_max:
                y_pad = 0.5 if y_min == 0 else abs(y_min) * 0.05
                plt.ylim(y_min - y_pad, y_max + y_pad)
            else:
                y_span = y_max - y_min
                y_pad = max(0.02 * y_span, 0.25)
                plt.ylim(y_min - y_pad, y_max + y_pad)
    except Exception:
        # Fallback to autoscale if anything goes wrong
        ax = plt.gca()
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if no_display:
        plt.close()
    else:
        plt.show()


def _compute_bodyfat_pct_from_weight(
    weights: List[float],
    dense_datetimes: List[datetime],
    baseline_weight_lb: float,
    baseline_lean_lb: float,
    lean_loss_fraction: float,
) -> List[float]:
    """
    Convert a time series of weights to body fat percent under an assumption that
    a fraction of the cumulative weight change from baseline comes from lean mass.

    L(t) = L0 + s * (W(t) - W0)  where s in [0,1]
    F(t) = W(t) - L(t)
    BF%(t) = 100 * F(t) / W(t)
    """
    bf_pct: List[float] = []
    W0 = float(baseline_weight_lb)
    L0 = float(baseline_lean_lb)
    s = float(lean_loss_fraction)
    for w in weights:
        w = float(w)
        # Compute lean mass under scenario
        lean = L0 + s * (w - W0)
        # Clamp physically meaningful bounds
        lean = max(0.0, min(lean, w))
        fat = max(0.0, w - lean)
        pct = 0.0 if w <= 0 else 100.0 * fat / w
        # Clamp to [0,100]
        pct = max(0.0, min(100.0, pct))
        bf_pct.append(pct)
    return bf_pct


def create_bodyfat_plot_from_kalman(
    entries,
    states,
    dates,
    baseline_weight_lb: Optional[float],
    baseline_lean_lb: float,
    output_path: str = "kalman_bodyfat_trend.png",
    no_display: bool = False,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    lbm_csv: Optional[str] = None,
    ci_multiplier: float = 1.96,
    enable_forecast: bool = True,
) -> None:
    """
    Create Estimated Body Fat % vs Date plot using Kalman-smoothed weights.

    - Lower bound: 0% lean mass loss (all change from fat)
    - Midline: 10% lean, 90% fat decomposition
    - Upper bound: 20% lean, 80% fat decomposition
    - Confidence band: propagate Kalman weight std to body fat for midline via
      evaluating W ± 1.96*std
    - Scatter points: per-entry body fat under 10%/90% assumption
    """
    if not entries or not states:
        return

    # Note: We don't filter entries here because Kalman algorithm needs full dataset
    # Date filtering will be applied to the dense grid later

    # Use smoothed Kalman mean and std over a dense date grid
    dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(states, dates)
    if not dense_datetimes:
        return
    
    # Filter dense curves by date range if specified
    if start_date or end_date:
        print(f"DEBUG: Filtering dense grid by date range: {start_date} to {end_date}")
        print(f"DEBUG: Original dense grid has {len(dense_datetimes)} points")
        print(f"DEBUG: Date range: {dense_datetimes[0].date()} to {dense_datetimes[-1].date()}")
        
        filtered_dense_datetimes = []
        filtered_dense_means = []
        filtered_dense_stds = []
        for dt, mean, std in zip(dense_datetimes, dense_means, dense_stds):
            dt_date = dt.date()
            if start_date and dt_date < start_date:
                continue
            if end_date and dt_date > end_date:
                continue
            filtered_dense_datetimes.append(dt)
            filtered_dense_means.append(mean)
            filtered_dense_stds.append(std)
        
        print(f"DEBUG: After filtering, dense grid has {len(filtered_dense_datetimes)} points")
        if not filtered_dense_datetimes:
            print(f"DEBUG: No data in specified date range after filtering!")
            return
            
        dense_datetimes = filtered_dense_datetimes
        dense_means = filtered_dense_means
        dense_stds = filtered_dense_stds

    # Baseline defaults: if baseline weight not provided, use first Kalman mean
    if baseline_weight_lb is None:
        baseline_weight_lb = float(dense_means[0])

    # If an LBM CSV is provided, drive body fat computations directly from LBM
    if lbm_csv:
        lbm_points = _load_lbm_csv(lbm_csv)
        if not lbm_points:
            # Fall back to original model if LBM file empty/unreadable
            lbm_csv = None

    if lbm_csv:
        # Build LBM series at dense times and entry times
        dense_lbm = _evaluate_lbm_series(dense_datetimes, lbm_points)
        # Body fat midline using provided LBM
        bf_mid = [float(max(0.0, min(100.0, 100.0 * (1.0 - (l / max(1e-6, w))))))
                  for w, l in zip(dense_means, dense_lbm)]

        # Confidence band: BF(W±z*std) with LBM fixed
        w_lo = [max(1e-6, m - ci_multiplier * s) for m, s in zip(dense_means, dense_stds)]
        w_hi = [m + ci_multiplier * s for m, s in zip(dense_means, dense_stds)]
        bf_mid_lo = [float(max(0.0, min(100.0, 100.0 * (1.0 - (l / max(1e-6, wl))))))
                     for wl, l in zip(w_lo, dense_lbm)]
        bf_mid_hi = [float(max(0.0, min(100.0, 100.0 * (1.0 - (l / max(1e-6, wh))))))
                     for wh, l in zip(w_hi, dense_lbm)]
        bf_band_lower = [min(a, b) for a, b in zip(bf_mid_lo, bf_mid_hi)]
        bf_band_upper = [max(a, b) for a, b in zip(bf_mid_lo, bf_mid_hi)]

        # Entry points BF from entries and interpolated LBM
        entry_datetimes = [e.entry_datetime for e in entries]
        entry_weights = [float(e.weight) for e in entries]
        entry_lbm = _evaluate_lbm_series(entry_datetimes, lbm_points)
        entry_bf_mid = [float(max(0.0, min(100.0, 100.0 * (1.0 - (l / max(1e-6, w))))))
                        for w, l in zip(entry_weights, entry_lbm)]
        show_scenarios = False
    else:
        # Scenario fractions
        s_low = 0.0
        s_mid = 0.10
        s_high = 0.20

        # Midline body fat from smoothed weights
        bf_mid = _compute_bodyfat_pct_from_weight(
            dense_means, dense_datetimes, baseline_weight_lb, baseline_lean_lb, s_mid
        )

        # Confidence band for midline: evaluate transformation at W±z*std
        w_lo = [max(1e-6, m - ci_multiplier * s) for m, s in zip(dense_means, dense_stds)]
        w_hi = [m + ci_multiplier * s for m, s in zip(dense_means, dense_stds)]
        bf_mid_lo = _compute_bodyfat_pct_from_weight(
            w_lo, dense_datetimes, baseline_weight_lb, baseline_lean_lb, s_mid
        )
        bf_mid_hi = _compute_bodyfat_pct_from_weight(
            w_hi, dense_datetimes, baseline_weight_lb, baseline_lean_lb, s_mid
        )
        # Ensure lower <= upper at each point
        bf_band_lower = [min(a, b) for a, b in zip(bf_mid_lo, bf_mid_hi)]
        bf_band_upper = [max(a, b) for a, b in zip(bf_mid_lo, bf_mid_hi)]

        # Scenario bounds using smoothed means
        bf_lower_bound = _compute_bodyfat_pct_from_weight(
            dense_means, dense_datetimes, baseline_weight_lb, baseline_lean_lb, s_low
        )
        bf_upper_bound = _compute_bodyfat_pct_from_weight(
            dense_means, dense_datetimes, baseline_weight_lb, baseline_lean_lb, s_high
        )

        # Scatter points for actual measurements under mid scenario
        entry_datetimes = [e.entry_datetime for e in entries]
        entry_weights = [float(e.weight) for e in entries]
        entry_bf_mid = _compute_bodyfat_pct_from_weight(
            entry_weights, entry_datetimes, baseline_weight_lb, baseline_lean_lb, s_mid
        )
        show_scenarios = True

    # Plot
    import matplotlib
    if no_display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8), dpi=100)

    # Confidence band around midline
    plt.fill_between(
        dense_datetimes, bf_band_lower, bf_band_upper,
        color="#cccccc", alpha=0.4, label="Midline 95% CI"
    )

    # Optional scenario bounds (only when not using external LBM)
    if show_scenarios:
        plt.plot(dense_datetimes, bf_lower_bound, "--", color="#2ca02c", linewidth=1.8,
                 label="Lower bound (0% lean loss)")
        plt.plot(dense_datetimes, bf_upper_bound, "--", color="#d62728", linewidth=1.8,
                 label="Upper bound (20% lean loss)")

    # Midline
    plt.plot(dense_datetimes, bf_mid, "-", color="#1f77b4", linewidth=2.4,
             label="Midline (10% lean, 90% fat)")

    # Scatter actual points under mid assumption (filter entries for display)
    if start_date or end_date:
        ed_filtered_dt = []
        ed_filtered_bf = []
        for dt_i, bf_i in zip(entry_datetimes, entry_bf_mid):
            d = dt_i.date()
            if start_date and d < start_date:
                continue
            if end_date and d > end_date:
                continue
            ed_filtered_dt.append(dt_i)
            ed_filtered_bf.append(bf_i)
        entry_datetimes_plot = ed_filtered_dt
        entry_bf_mid_plot = ed_filtered_bf
    else:
        entry_datetimes_plot = entry_datetimes
        entry_bf_mid_plot = entry_bf_mid

    plt.scatter(entry_datetimes_plot, entry_bf_mid_plot, s=36, color="#1f77b4", alpha=0.8,
                edgecolors="white", linewidths=0.5, zorder=5, label="Mid assumption (measurements)")

    # Stats panel: current estimate, rate, and forecasts (1wk, 1mo)
    # Current estimate at latest date (match curve): use bf_mid[-1]
    latest_date = dates[-1]
    current_bf = float(bf_mid[-1])
    # Approximate current CI from band half-width at last index
    current_halfwidth = float(max(bf_band_upper[-1] - current_bf, current_bf - bf_band_lower[-1]))

    # Current rate (bf% per week) from derivative of a spline on bf_mid
    try:
        from scipy.interpolate import CubicSpline  # type: ignore
        t0_dt = dense_datetimes[0]
        t_days = np.array([(dt - t0_dt).total_seconds() / 86400.0 for dt in dense_datetimes], dtype=float)
        bf_mid_arr = np.array(bf_mid, dtype=float)
        bf_spline = CubicSpline(t_days, bf_mid_arr, bc_type="natural")
        bf_slope_per_day = float(bf_spline.derivative()(t_days[-1]))
    except Exception:
        # Fallback: finite difference over last two points
        t0_dt = dense_datetimes[0]
        t_days = np.array([(dt - t0_dt).total_seconds() / 86400.0 for dt in dense_datetimes], dtype=float)
        if len(t_days) >= 2:
            dt_last = float(t_days[-1] - t_days[-2]) if t_days[-1] != t_days[-2] else 1.0
            bf_slope_per_day = float((bf_mid[-1] - bf_mid[-2]) / dt_last)
        else:
            bf_slope_per_day = 0.0
    bf_slope_per_week = 7.0 * bf_slope_per_day
    
    # Calculate body fat rate uncertainty by propagating weight velocity uncertainty
    # For the body fat model: BF% = 100 * (W - L) / W where L = L0 + s*(W - W0)
    # The rate uncertainty comes from the weight velocity uncertainty
    weight_velocity_std = np.sqrt(max(states[-1].velocity_var, 0.0))
    weight_velocity_ci = ci_multiplier * weight_velocity_std
    
    # Approximate BF rate uncertainty using finite differences around current weight
    current_weight = float(dense_means[-1])
    weight_perturbation = weight_velocity_ci * 7.0  # 1 week worth of velocity uncertainty
    
    if lbm_csv:
        # For LBM-based calculation, uncertainty comes from weight uncertainty
        current_lbm = _evaluate_lbm_series([dense_datetimes[-1]], lbm_points)[0]
        bf_current = float(max(0.0, min(100.0, 100.0 * (1.0 - (current_lbm / max(1e-6, current_weight))))))
        bf_pert_lo = float(max(0.0, min(100.0, 100.0 * (1.0 - (current_lbm / max(1e-6, current_weight - weight_perturbation))))))
        bf_pert_hi = float(max(0.0, min(100.0, 100.0 * (1.0 - (current_lbm / max(1e-6, current_weight + weight_perturbation))))))
    else:
        # For model-based calculation with s_mid = 0.10
        s_mid = 0.10
        bf_current = _compute_bodyfat_pct_from_weight([current_weight], [dense_datetimes[-1]], baseline_weight_lb, baseline_lean_lb, s_mid)[0]
        bf_pert_lo = _compute_bodyfat_pct_from_weight([current_weight - weight_perturbation], [dense_datetimes[-1]], baseline_weight_lb, baseline_lean_lb, s_mid)[0]
        bf_pert_hi = _compute_bodyfat_pct_from_weight([current_weight + weight_perturbation], [dense_datetimes[-1]], baseline_weight_lb, baseline_lean_lb, s_mid)[0]
    
    # BF rate uncertainty (per week)
    bf_rate_uncertainty = max(abs(bf_pert_hi - bf_current), abs(bf_current - bf_pert_lo))

    # Forecasts: transform Kalman weight forecasts through BF model
    from kalman import WeightKalmanFilter
    kf = WeightKalmanFilter(
        initial_weight=states[-1].weight,
        initial_velocity=states[-1].velocity,
        initial_weight_var=states[-1].weight_var,
        initial_velocity_var=states[-1].velocity_var,
    )
    kf.x = np.array([states[-1].weight, states[-1].velocity], dtype=float)
    kf.P = np.array([
        [states[-1].weight_var, states[-1].weight_velocity_cov],
        [states[-1].weight_velocity_cov, states[-1].velocity_var],
    ], dtype=float)

    # Helper to compute BF at a future offset
    def _bf_from_weight_at_offset(days_fwd: float, w: float, w_std: float) -> Tuple[float, float]:
        future_dt = dense_datetimes[-1] + timedelta(days=days_fwd)
        if lbm_csv:
            future_lbm = _evaluate_lbm_series([future_dt], lbm_points)[0]
            bf_mid_val = float(max(0.0, min(100.0, 100.0 * (1.0 - (future_lbm / max(1e-6, w))))))
            w_lo_f = max(1e-6, w - ci_multiplier * w_std)
            w_hi_f = w + ci_multiplier * w_std
            bf_lo_val = float(max(0.0, min(100.0, 100.0 * (1.0 - (future_lbm / max(1e-6, w_lo_f))))))
            bf_hi_val = float(max(0.0, min(100.0, 100.0 * (1.0 - (future_lbm / max(1e-6, w_hi_f))))))
        else:
            s_mid = 0.10
            bf_mid_val = _compute_bodyfat_pct_from_weight([w], [future_dt], baseline_weight_lb, baseline_lean_lb, s_mid)[0]
            bf_lo_val = _compute_bodyfat_pct_from_weight([max(1e-6, w - ci_multiplier * w_std)], [future_dt], baseline_weight_lb, baseline_lean_lb, s_mid)[0]
            bf_hi_val = _compute_bodyfat_pct_from_weight([w + ci_multiplier * w_std], [future_dt], baseline_weight_lb, baseline_lean_lb, s_mid)[0]
        halfwidth = float(max(bf_hi_val - bf_mid_val, bf_mid_val - bf_lo_val))
        return bf_mid_val, halfwidth

    # Add forecasting fan if enabled
    if enable_forecast:  # Show forecast even when filtering by date
        # Create forecast dates (1 month = 30 days) - start from day 0 to eliminate gap
        forecast_days = 30
        last_date = dense_datetimes[-1]
        forecast_dates = [last_date + timedelta(days=i) for i in range(0, forecast_days + 1)]
        
        # Calculate forecast body fat values using the existing Kalman filter
        forecast_bf_mid = []
        forecast_bf_upper = []
        forecast_bf_lower = []
        
        for i in range(0, forecast_days + 1):
            w_forecast, w_forecast_std = kf.forecast(float(i))
            bf_mid_val, bf_halfwidth = _bf_from_weight_at_offset(float(i), w_forecast, w_forecast_std)
            forecast_bf_mid.append(bf_mid_val)
            forecast_bf_upper.append(bf_mid_val + bf_halfwidth)
            forecast_bf_lower.append(bf_mid_val - bf_halfwidth)
        
        # Plot forecast fan
        ci_label = "95%" if abs(ci_multiplier - 1.96) < 0.01 else "1σ"
        plt.fill_between(forecast_dates, forecast_bf_lower, forecast_bf_upper, 
                        alpha=0.2, color='gray')
        
        # Plot forecast mean line - use same color as main plot line (#1f77b4)
        plt.plot(forecast_dates, forecast_bf_mid, '--', color='#1f77b4', linewidth=1.5, 
                alpha=0.8, label='Body Fat Forecast')

    # 1 week forecast
    w_week, w_week_std = kf.forecast(7.0)
    bf_week_mid, bf_week_halfwidth = _bf_from_weight_at_offset(7.0, w_week, w_week_std)

    # 1 month forecast (30 days)
    w_month, w_month_std = kf.forecast(30.0)
    bf_month_mid, bf_month_halfwidth = _bf_from_weight_at_offset(30.0, w_month, w_month_std)

    # Stats box text
    stats_text = (
        f"Current Estimate ({latest_date})\n"
        f"Body Fat: {current_bf:.2f}% ± {current_halfwidth:.2f}%\n"
        f"Rate: {bf_slope_per_week:+.3f} ± {bf_rate_uncertainty:.3f}%/week\n\n"
        f"Forecasts:\n"
        f"1 week: {bf_week_mid:.2f}% ± {bf_week_halfwidth:.2f}%\n"
        f"1 month: {bf_month_mid:.2f}% ± {bf_month_halfwidth:.2f}%"
    )

    ax = plt.gca()
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white',
                     alpha=0.9, edgecolor='#cccccc'), zorder=10)

    plt.title("Estimated Body Fat % (Kalman)")
    plt.xlabel("Date")
    plt.ylabel("Body Fat (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set x-axis using filtered dense range with padding so bands stay inside
    dense_min = min(dense_datetimes) if dense_datetimes else None
    dense_max = max(dense_datetimes) if dense_datetimes else None
    if start_date or end_date:
        left = dt.combine(start_date, dt.min.time()) if start_date else dense_min
        right = dt.combine(end_date, dt.max.time()) if end_date else dense_max
    else:
        left, right = dense_min, dense_max
    if left is not None and right is not None and left <= right:
        span = right - left
        pad = max(span * 0.02, timedelta(days=1))
        left_pad = left if start_date else (left - pad)
        # Always extend 30 days into the future, regardless of date filtering
        right_pad = right + timedelta(days=30) if end_date else (right + timedelta(days=30))
        plt.xlim(left=left_pad, right=right_pad)
    
    # After setting x-limits, compute Y-limits from visible series
    try:
        y_values: List[float] = []
        # Confidence band and midline
        y_values.extend([float(v) for v in bf_band_lower])
        y_values.extend([float(v) for v in bf_band_upper])
        y_values.extend([float(v) for v in bf_mid])
        # Optional scenario bounds
        if show_scenarios:
            y_values.extend([float(v) for v in bf_lower_bound])
            y_values.extend([float(v) for v in bf_upper_bound])
        # Scatter points
        y_values.extend([float(v) for v in entry_bf_mid_plot])
        # Include forecast values in y-axis range calculation
        if enable_forecast and 'forecast_bf_mid' in locals():
            y_values.extend([float(v) for v in forecast_bf_mid])
            y_values.extend([float(v) for v in forecast_bf_upper])
            y_values.extend([float(v) for v in forecast_bf_lower])
        if y_values:
            y_min = float(np.nanmin(np.array(y_values, dtype=float)))
            y_max = float(np.nanmax(np.array(y_values, dtype=float)))
            if y_min == y_max:
                y_pad = 0.5 if y_min == 0 else abs(y_min) * 0.05
                plt.ylim(y_min - y_pad, y_max + y_pad)
            else:
                y_span = y_max - y_min
                y_pad = max(0.02 * y_span, 0.25)
                plt.ylim(y_min - y_pad, y_max + y_pad)
    except Exception:
        ax = plt.gca()
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if no_display:
        plt.close()
    else:
        plt.show()


def compute_residuals(entries, states, dates, start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[float]:
    """
    Compute residuals (differences) between Kalman filter output and raw measurements.
    
    Args:
        entries: List of WeightEntry objects
        states: List of KalmanState objects from Kalman filter
        dates: List of dates corresponding to states
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        
    Returns:
        List of residuals (raw_weight - kalman_weight) for each measurement
    """
    if not entries or not states or not dates:
        return []
    
    # Filter entries by date range if specified
    filtered_entries = entries
    if start_date or end_date:
        from datetime import datetime as dt, timedelta
        if start_date:
            # Convert start_date to datetime at beginning of day
            start_datetime = dt.combine(start_date, dt.min.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime >= start_datetime]
        if end_date:
            # Convert end_date to datetime at end of day
            end_datetime = dt.combine(end_date, dt.max.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime <= end_datetime]
        
        if not filtered_entries:
            return []
    
    # Create a mapping from datetime to Kalman state for efficient lookup
    datetime_to_state = {d: s for d, s in zip(dates, states)}
    
    residuals = []
    for entry in filtered_entries:
        if entry.entry_datetime in datetime_to_state:
            kalman_weight = datetime_to_state[entry.entry_datetime].weight
            residual = entry.weight - kalman_weight
            residuals.append(residual)
    
    return residuals


def _load_height_data(height_file: str) -> Optional[float]:
    """Load height data from height.txt file. Returns height in inches or None if not found."""
    try:
        if os.path.exists(height_file):
            with open(height_file, 'r') as f:
                height_inches = float(f.read().strip())
                return height_inches
    except Exception:
        pass
    return None


def create_bmi_plot_from_kalman(
    entries,
    states,
    dates,
    height_file: str = "data/height.txt",
    output_path: str = "bmi_trend.png",
    no_display: bool = False,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    ci_multiplier: float = 1.96,
    enable_forecast: bool = True,
) -> None:
    """
    Create BMI vs Date plot using Kalman-smoothed weights and height data.
    
    BMI = weight(kg) / height(m)^2
    For imperial units: BMI = (weight_lb / height_in^2) * 703
    """
    from datetime import datetime as dt, timedelta
    
    if not entries or not states:
        return

    # Load height data
    height_inches = _load_height_data(height_file)
    if height_inches is None:
        print(f"Warning: No height data found at {height_file}. BMI plot cannot be generated.")
        return

    # Note: We don't filter entries here because Kalman algorithm needs full dataset
    # Date filtering will be applied to the dense grid later

    # Use smoothed Kalman mean and std over a dense date grid
    dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(states, dates)
    if not dense_datetimes:
        return
    
    # Filter dense curves by date range if specified
    if start_date or end_date:
        print(f"DEBUG BMI: Filtering dense grid by date range: {start_date} to {end_date}")
        print(f"DEBUG BMI: Original dense grid has {len(dense_datetimes)} points")
        print(f"DEBUG BMI: Date range: {dense_datetimes[0].date()} to {dense_datetimes[-1].date()}")
        
        filtered_dense_datetimes = []
        filtered_dense_means = []
        filtered_dense_stds = []
        for dt, mean, std in zip(dense_datetimes, dense_means, dense_stds):
            dt_date = dt.date()
            if start_date and dt_date < start_date:
                continue
            if end_date and dt_date > end_date:
                continue
            filtered_dense_datetimes.append(dt)
            filtered_dense_means.append(mean)
            filtered_dense_stds.append(std)
        
        print(f"DEBUG BMI: After filtering, dense grid has {len(filtered_dense_datetimes)} points")
        if not filtered_dense_datetimes:
            print(f"DEBUG BMI: No data in specified date range after filtering!")
            return
            
        dense_datetimes = filtered_dense_datetimes
        dense_means = filtered_dense_means
        dense_stds = filtered_dense_stds

    # Convert height to meters for BMI calculation
    height_m = height_inches * 0.0254  # inches to meters
    
    # Calculate BMI from Kalman-smoothed weights
    # BMI = weight(kg) / height(m)^2
    # weight_lb * 0.453592 = weight_kg
    bmi_means = [(w * 0.453592) / (height_m ** 2) for w in dense_means]
    
    # Calculate BMI confidence band
    weight_lo = [max(1e-6, m - ci_multiplier * s) for m, s in zip(dense_means, dense_stds)]
    weight_hi = [m + ci_multiplier * s for m, s in zip(dense_means, dense_stds)]
    bmi_lo = [(w * 0.453592) / (height_m ** 2) for w in weight_lo]
    bmi_hi = [(w * 0.453592) / (height_m ** 2) for w in weight_hi]
    
    # Calculate BMI for actual measurements
    entry_datetimes = [e.entry_datetime for e in entries]
    entry_weights = [float(e.weight) for e in entries]
    entry_bmi = [(w * 0.453592) / (height_m ** 2) for w in entry_weights]

    # Plot
    import matplotlib
    if no_display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8), dpi=100)

    # Confidence band around BMI mean
    plt.fill_between(
        dense_datetimes,
        bmi_lo,
        bmi_hi,
        alpha=0.3,
        color="blue",
        label=f"BMI {ci_multiplier:.1f}σ confidence band"
    )

    # BMI mean line
    plt.plot(dense_datetimes, bmi_means, "b-", linewidth=2, label="BMI (Kalman smoothed)")

    # Scatter points for actual measurements (filter for display if date range set)
    if start_date or end_date:
        ed_filtered_dt = []
        ed_filtered_bmi = []
        for dt_i, bmi_i in zip(entry_datetimes, entry_bmi):
            d = dt_i.date()
            if start_date and d < start_date:
                continue
            if end_date and d > end_date:
                continue
            ed_filtered_dt.append(dt_i)
            ed_filtered_bmi.append(bmi_i)
        entry_datetimes_plot = ed_filtered_dt
        entry_bmi_plot = ed_filtered_bmi
    else:
        entry_datetimes_plot = entry_datetimes
        entry_bmi_plot = entry_bmi

    plt.scatter(entry_datetimes_plot, entry_bmi_plot, color="red", s=30, alpha=0.7, label="BMI (measurements)")

    # BMI categories
    plt.axhline(y=18.5, color="green", linestyle="--", alpha=0.7, label="Underweight (18.5)")
    plt.axhline(y=25.0, color="orange", linestyle="--", alpha=0.7, label="Normal (25.0)")
    plt.axhline(y=30.0, color="red", linestyle="--", alpha=0.7, label="Overweight (30.0)")

    # Add forecasting fan if enabled
    if enable_forecast:  # Show forecast even when filtering by date
        # Get the last state for forecasting
        last_state = states[-1]
        last_date = dates[-1]
        
        # Calculate current BMI from last state
        last_bmi = (last_state.weight * 0.453592) / (height_m ** 2)
        last_bmi_std = (np.sqrt(last_state.weight_var) * 0.453592) / (height_m ** 2)
        
        # Create forecast dates (1 month = 30 days) - start from day 0 to eliminate gap
        forecast_days = 30
        forecast_dates = [last_date + timedelta(days=i) for i in range(0, forecast_days + 1)]
        
        # For BMI forecasting, we need to estimate weight trend
        # Use the velocity from the last state to project weight forward
        weight_velocity = last_state.velocity  # lbs/day
        
        # Calculate forecast weights and convert to BMI
        forecast_weights = [last_state.weight + weight_velocity * i for i in range(0, forecast_days + 1)]
        forecast_bmi = [(w * 0.453592) / (height_m ** 2) for w in forecast_weights]
        
        # Calculate uncertainty bands (growing uncertainty over time)
        # Use the velocity variance to estimate growing uncertainty
        velocity_std = np.sqrt(last_state.velocity_var)
        forecast_weight_stds = [np.sqrt(last_state.weight_var + (velocity_std * i) ** 2) for i in range(0, forecast_days + 1)]
        forecast_bmi_stds = [(std * 0.453592) / (height_m ** 2) for std in forecast_weight_stds]
        
        # Create confidence intervals
        forecast_upper = [bmi + ci_multiplier * std for bmi, std in zip(forecast_bmi, forecast_bmi_stds)]
        forecast_lower = [bmi - ci_multiplier * std for bmi, std in zip(forecast_bmi, forecast_bmi_stds)]
        
        # Plot forecast fan
        ci_label = "95%" if abs(ci_multiplier - 1.96) < 0.01 else "1σ"
        plt.fill_between(forecast_dates, forecast_lower, forecast_upper, 
                        alpha=0.2, color='gray')
        
        # Plot forecast mean line - use same color as main plot line (blue)
        plt.plot(forecast_dates, forecast_bmi, '--', color='blue', linewidth=1.5, 
                alpha=0.8, label='BMI Forecast')

    plt.xlabel("Date")
    plt.ylabel("BMI (kg/m²)")
    plt.title(f"BMI Trend (Height: {height_inches:.1f}\" = {height_m:.2f}m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Set x-axis using filtered dense range with padding so bands stay inside
    dense_min = min(dense_datetimes) if dense_datetimes else None
    dense_max = max(dense_datetimes) if dense_datetimes else None
    if start_date or end_date:
        left = dt.combine(start_date, dt.min.time()) if start_date else dense_min
        right = dt.combine(end_date, dt.max.time()) if end_date else dense_max
    else:
        left, right = dense_min, dense_max
    if left is not None and right is not None and left <= right:
        span = right - left
        pad = max(span * 0.02, timedelta(days=1))
        left_pad = left if start_date else (left - pad)
        # Always extend 30 days into the future, regardless of date filtering
        right_pad = right + timedelta(days=30) if end_date else (right + timedelta(days=30))
        plt.xlim(left=left_pad, right=right_pad)
    
    # After setting x-limits, compute Y-limits from visible series
    try:
        y_values: List[float] = []
        # Confidence band and mean line
        y_values.extend([float(v) for v in bmi_lo])
        y_values.extend([float(v) for v in bmi_hi])
        y_values.extend([float(v) for v in bmi_means])
        # Scatter points
        y_values.extend([float(v) for v in entry_bmi_plot])
        # Include forecast values in y-axis range calculation
        if enable_forecast and 'forecast_bmi' in locals():
            y_values.extend([float(v) for v in forecast_bmi])
            y_values.extend([float(v) for v in forecast_upper])
            y_values.extend([float(v) for v in forecast_lower])
        if y_values:
            y_min = float(np.nanmin(np.array(y_values, dtype=float)))
            y_max = float(np.nanmax(np.array(y_values, dtype=float)))
            if y_min == y_max:
                y_pad = 0.5 if y_min == 0 else abs(y_min) * 0.05
                plt.ylim(y_min - y_pad, y_max + y_pad)
            else:
                y_span = y_max - y_min
                y_pad = max(0.02 * y_span, 0.25)
                plt.ylim(y_min - y_pad, y_max + y_pad)
    except Exception:
        ax = plt.gca()
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)

    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if not no_display:
        plt.show()
    plt.close()


def create_ffmi_plot_from_kalman(
    entries,
    states,
    dates,
    height_file: str = "data/height.txt",
    output_path: str = "ffmi_trend.png",
    no_display: bool = False,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    lbm_csv: Optional[str] = None,
    ci_multiplier: float = 1.96,
    enable_forecast: bool = True,
) -> None:
    """
    Create FFMI (Fat-Free Mass Index) vs Date plot using Kalman-smoothed weights and height data.
    
    FFMI = LBM(kg) / height(m)^2
    For imperial units: FFMI = (LBM_lb / height_in^2) * 703
    """
    from datetime import datetime as dt, timedelta
    
    if not entries or not states:
        return

    # Load height data
    height_inches = _load_height_data(height_file)
    if height_inches is None:
        print(f"Warning: No height data found at {height_file}. FFMI plot cannot be generated.")
        return

    # Note: We don't filter entries here because Kalman algorithm needs full dataset
    # Date filtering will be applied to the dense grid later

    # Use smoothed Kalman mean and std over a dense date grid
    dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(states, dates)
    if not dense_datetimes:
        return
    
    # Filter dense curves by date range if specified
    if start_date or end_date:
        print(f"DEBUG FFMI: Filtering dense grid by date range: {start_date} to {end_date}")
        print(f"DEBUG FFMI: Original dense grid has {len(dense_datetimes)} points")
        print(f"DEBUG FFMI: Date range: {dense_datetimes[0].date()} to {dense_datetimes[-1].date()}")
        
        filtered_dense_datetimes = []
        filtered_dense_means = []
        filtered_dense_stds = []
        for dt, mean, std in zip(dense_datetimes, dense_means, dense_stds):
            dt_date = dt.date()
            if start_date and dt_date < start_date:
                continue
            if end_date and dt_date > end_date:
                continue
            filtered_dense_datetimes.append(dt)
            filtered_dense_means.append(mean)
            filtered_dense_stds.append(std)
        
        print(f"DEBUG FFMI: After filtering, dense grid has {len(filtered_dense_datetimes)} points")
        if not filtered_dense_datetimes:
            print(f"DEBUG FFMI: No data in specified date range after filtering!")
            return
            
        dense_datetimes = filtered_dense_datetimes
        dense_means = filtered_dense_means
        dense_stds = filtered_dense_stds

    # Convert height to meters for FFMI calculation
    height_m = height_inches * 0.0254  # inches to meters
    
    # Calculate LBM from Kalman-smoothed weights
    # Use the same LBM calculation logic as body fat plot
    if lbm_csv:
        lbm_points = _load_lbm_csv(lbm_csv)
        if not lbm_points:
            # Fall back to estimated LBM if LBM file empty/unreadable
            lbm_points = None
    else:
        lbm_points = None
    
    if lbm_points:
        # Use provided LBM data
        dense_lbm = _evaluate_lbm_series(dense_datetimes, lbm_points)
        # Calculate FFMI from LBM
        ffmi_means = [(lbm * 0.453592) / (height_m ** 2) for lbm in dense_lbm]
        
        # Calculate FFMI confidence band using weight uncertainty
        # Approximate LBM uncertainty from weight uncertainty (assuming constant body fat %)
        weight_lo = [max(1e-6, m - ci_multiplier * s) for m, s in zip(dense_means, dense_stds)]
        weight_hi = [m + ci_multiplier * s for m, s in zip(dense_means, dense_stds)]
        # For simplicity, assume LBM varies proportionally with weight
        lbm_lo = [lbm * (w_lo / w) for lbm, w_lo, w in zip(dense_lbm, weight_lo, dense_means)]
        lbm_hi = [lbm * (w_hi / w) for lbm, w_hi, w in zip(dense_lbm, weight_hi, dense_means)]
        ffmi_lo = [(lbm * 0.453592) / (height_m ** 2) for lbm in lbm_lo]
        ffmi_hi = [(lbm * 0.453592) / (height_m ** 2) for lbm in lbm_hi]
        
        # Calculate FFMI for actual measurements
        entry_datetimes = [e.entry_datetime for e in entries]
        entry_weights = [float(e.weight) for e in entries]
        entry_lbm = _evaluate_lbm_series(entry_datetimes, lbm_points)
        entry_ffmi = [(lbm * 0.453592) / (height_m ** 2) for lbm in entry_lbm]
    else:
        # Estimate LBM from weight using a simple model (assuming ~15% body fat)
        estimated_bf = 0.15  # 15% body fat
        dense_lbm = [w * (1.0 - estimated_bf) for w in dense_means]
        ffmi_means = [(lbm * 0.453592) / (height_m ** 2) for lbm in dense_lbm]
        
        # Calculate FFMI confidence band
        weight_lo = [max(1e-6, m - ci_multiplier * s) for m, s in zip(dense_means, dense_stds)]
        weight_hi = [m + ci_multiplier * s for m, s in zip(dense_means, dense_stds)]
        lbm_lo = [w * (1.0 - estimated_bf) for w in weight_lo]
        lbm_hi = [w * (1.0 - estimated_bf) for w in weight_hi]
        ffmi_lo = [(lbm * 0.453592) / (height_m ** 2) for lbm in lbm_lo]
        ffmi_hi = [(lbm * 0.453592) / (height_m ** 2) for lbm in lbm_hi]
        
        # Calculate FFMI for actual measurements
        entry_datetimes = [e.entry_datetime for e in entries]
        entry_weights = [float(e.weight) for e in entries]
        entry_lbm = [w * (1.0 - estimated_bf) for w in entry_weights]
        entry_ffmi = [(lbm * 0.453592) / (height_m ** 2) for lbm in entry_lbm]

    # Plot
    import matplotlib
    if no_display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8), dpi=100)

    # Confidence band around FFMI mean
    plt.fill_between(
        dense_datetimes,
        ffmi_lo,
        ffmi_hi,
        alpha=0.3,
        color="blue",
        label=f"FFMI {ci_multiplier:.1f}σ confidence band"
    )

    # FFMI mean line
    plt.plot(dense_datetimes, ffmi_means, "b-", linewidth=2, label="FFMI (Kalman smoothed)")

    # Scatter points for actual measurements (filter for display if date range set)
    if start_date or end_date:
        ed_filtered_dt = []
        ed_filtered_ffmi = []
        for dt_i, ffmi_i in zip(entry_datetimes, entry_ffmi):
            d = dt_i.date()
            if start_date and d < start_date:
                continue
            if end_date and d > end_date:
                continue
            ed_filtered_dt.append(dt_i)
            ed_filtered_ffmi.append(ffmi_i)
        entry_datetimes_plot = ed_filtered_dt
        entry_ffmi_plot = ed_filtered_ffmi
    else:
        entry_datetimes_plot = entry_datetimes
        entry_ffmi_plot = entry_ffmi

    plt.scatter(entry_datetimes_plot, entry_ffmi_plot, color="red", s=30, alpha=0.7, label="FFMI (measurements)")

    # FFMI reference lines (typical ranges for men)
    plt.axhline(y=16.0, color="red", linestyle="--", alpha=0.7, label="Below average (16.0)")
    plt.axhline(y=18.0, color="orange", linestyle="--", alpha=0.7, label="Average (18.0)")
    plt.axhline(y=20.0, color="green", linestyle="--", alpha=0.7, label="Above average (20.0)")
    plt.axhline(y=22.0, color="blue", linestyle="--", alpha=0.7, label="Excellent (22.0)")

    # Add forecasting fan if enabled
    if enable_forecast:  # Show forecast even when filtering by date
        # Get the last state for forecasting
        last_state = states[-1]
        last_date = dates[-1]
        
        # Create forecast dates (1 month = 30 days) - start from day 0 to eliminate gap
        forecast_days = 30
        forecast_dates = [last_date + timedelta(days=i) for i in range(0, forecast_days + 1)]
        
        # For FFMI forecasting, we need to estimate weight and LBM trends
        # Use the velocity from the last state to project weight forward
        weight_velocity = last_state.velocity  # lbs/day
        
        # Calculate forecast weights
        forecast_weights = [last_state.weight + weight_velocity * i for i in range(0, forecast_days + 1)]
        
        # Calculate forecast LBM (assuming 10% lean mass loss rate)
        s_mid = 0.10  # 10% lean, 90% fat decomposition
        if lbm_csv:
            # Use external LBM data if available
            lbm_points = _load_lbm_csv(lbm_csv)
            forecast_lbm = [_evaluate_lbm_series([last_date + timedelta(days=i)], lbm_points)[0] for i in range(0, forecast_days + 1)]
        else:
            # Use model-based LBM calculation
            baseline_weight = last_state.weight  # Use current weight as baseline
            baseline_lean = last_state.weight * 0.7  # Assume 70% lean mass initially
            forecast_lbm = []
            for i in range(0, forecast_days + 1):
                w = forecast_weights[i]
                # LBM = L0 + s*(W - W0) where s = 0.10
                lbm = baseline_lean + s_mid * (w - baseline_weight)
                forecast_lbm.append(max(0, lbm))
        
        # Calculate forecast FFMI
        forecast_ffmi = [(lbm * 0.453592) / (height_m ** 2) for lbm in forecast_lbm]
        
        # Calculate uncertainty bands (growing uncertainty over time)
        velocity_std = np.sqrt(last_state.velocity_var)
        forecast_weight_stds = [np.sqrt(last_state.weight_var + (velocity_std * i) ** 2) for i in range(0, forecast_days + 1)]
        
        # Convert weight uncertainty to FFMI uncertainty
        # FFMI = LBM(kg) / height(m)^2, where LBM depends on weight
        # For simplicity, assume FFMI uncertainty scales with weight uncertainty
        forecast_ffmi_stds = [(std * 0.453592 * s_mid) / (height_m ** 2) for std in forecast_weight_stds]
        
        # Create confidence intervals
        forecast_upper = [ffmi + ci_multiplier * std for ffmi, std in zip(forecast_ffmi, forecast_ffmi_stds)]
        forecast_lower = [ffmi - ci_multiplier * std for ffmi, std in zip(forecast_ffmi, forecast_ffmi_stds)]
        
        # Plot forecast fan
        ci_label = "95%" if abs(ci_multiplier - 1.96) < 0.01 else "1σ"
        plt.fill_between(forecast_dates, forecast_lower, forecast_upper, 
                        alpha=0.2, color='gray')
        
        # Plot forecast mean line - use same color as main plot line (blue)
        plt.plot(forecast_dates, forecast_ffmi, '--', color='blue', linewidth=1.5, 
                alpha=0.8, label='FFMI Forecast')

    plt.xlabel("Date")
    plt.ylabel("FFMI (kg/m²)")
    plt.title(f"FFMI Trend (Height: {height_inches:.1f}\" = {height_m:.2f}m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Set x-axis limits based on date filtering
    if start_date or end_date:
        min_datetime = min(dates) if dates else None
        max_datetime = max(dates) if dates else None
        if start_date:
            start_datetime = dt.combine(start_date, dt.min.time())
            plt.xlim(left=start_datetime)
        elif min_datetime is not None:
            plt.xlim(left=min_datetime)
        if end_date:
            end_datetime = dt.combine(end_date, dt.max.time())
            # Always extend 30 days into the future, regardless of date filtering
            plt.xlim(right=end_datetime + timedelta(days=30))
        elif max_datetime is not None:
            # Always extend 30 days into the future, regardless of date filtering
            plt.xlim(right=max_datetime + timedelta(days=30))
    else:
        # Default to full data range when no explicit start/end dates are provided
        if dates:
            min_datetime = min(dates)
            max_datetime = max(dates)
            # Always extend 30 days into the future, regardless of date filtering
            plt.xlim(left=min_datetime, right=max_datetime + timedelta(days=30))
    
    # After setting x-limits, compute Y-limits from visible series
    try:
        y_values: List[float] = []
        # Confidence band and mean line
        y_values.extend([float(v) for v in ffmi_lo])
        y_values.extend([float(v) for v in ffmi_hi])
        y_values.extend([float(v) for v in ffmi_means])
        # Scatter points
        y_values.extend([float(v) for v in entry_ffmi_plot])
        # Include forecast values in y-axis range calculation
        if enable_forecast and 'forecast_ffmi' in locals():
            y_values.extend([float(v) for v in forecast_ffmi])
            y_values.extend([float(v) for v in forecast_upper])
            y_values.extend([float(v) for v in forecast_lower])
        if y_values:
            y_min = float(np.nanmin(np.array(y_values, dtype=float)))
            y_max = float(np.nanmax(np.array(y_values, dtype=float)))
            if y_min == y_max:
                y_pad = 0.5 if y_min == 0 else abs(y_min) * 0.05
                plt.ylim(y_min - y_pad, y_max + y_pad)
            else:
                y_span = y_max - y_min
                y_pad = max(0.02 * y_span, 0.25)
                plt.ylim(y_min - y_pad, y_max + y_pad)
    except Exception:
        ax = plt.gca()
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)

    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if not no_display:
        plt.show()
    plt.close()


def create_residuals_histogram(residuals: List[float], 
                              output_path: str = "residuals_histogram.png",
                              no_display: bool = False,
                              start_date: Optional[date] = None,
                              end_date: Optional[date] = None,
                              ci_multiplier: float = 1.96) -> Tuple[float, float, float]:
    """
    Create a normalized histogram of residuals with normal distribution overlay and normality test.
    
    Args:
        residuals: List of residuals (raw_weight - kalman_weight)
        output_path: Path to save the plot
        no_display: Whether to display the plot or just save it
        start_date: Optional start date for plot title
        end_date: Optional end date for plot title
        
    Returns:
        Tuple of (mean, std, p_value) from normality test
    """
    if not residuals:
        print("No residuals data available for histogram")
        return 0.0, 0.0, 1.0
    
    residuals_array = np.array(residuals)
    mean_residual = np.mean(residuals_array)
    std_residual = np.std(residuals_array, ddof=1)  # Sample standard deviation
    skewness = stats.skew(residuals_array)
    excess_kurtosis = stats.kurtosis(residuals_array)  # Excess kurtosis (normal = 0)
    
    # Perform Shapiro-Wilk normality test
    if len(residuals_array) >= 3:  # Minimum sample size for Shapiro-Wilk
        shapiro_stat, shapiro_p = stats.shapiro(residuals_array)
        # Also try Kolmogorov-Smirnov test as backup
        ks_stat, ks_p = stats.kstest(residuals_array, 'norm', args=(mean_residual, std_residual))
    else:
        shapiro_stat, shapiro_p = 0.0, 1.0
        ks_stat, ks_p = 0.0, 1.0
    
    # Create the plot
    import matplotlib
    if no_display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    # Create histogram with density=True for normalization
    n, bins, patches = plt.hist(residuals_array, bins=20, density=True, alpha=0.7, 
                               color='skyblue', edgecolor='black', linewidth=0.5,
                               label=f'Residuals (n={len(residuals_array)})')
    
    # Create normal distribution overlay with same mean and std
    x_range = np.linspace(residuals_array.min(), residuals_array.max(), 100)
    normal_dist = stats.norm.pdf(x_range, loc=0, scale=std_residual)  # Mean assumed to be 0
    plt.plot(x_range, normal_dist, 'r-', linewidth=2, 
             label=f'Normal(μ=0, σ={std_residual:.3f})')
    
    # Add vertical line at mean
    plt.axvline(mean_residual, color='green', linestyle='--', linewidth=2, 
                label=f'Mean = {mean_residual:.3f}')
    
    # Add vertical lines at ±1 and ±2 standard deviations
    plt.axvline(std_residual, color='orange', linestyle=':', alpha=0.7, 
                label=f'±1σ = {std_residual:.3f}')
    plt.axvline(ci_multiplier*std_residual, color='red', linestyle=':', alpha=0.7, 
                label=f'+{ci_multiplier:.1f}σ = {ci_multiplier*std_residual:.3f}')
    plt.axvline(-ci_multiplier*std_residual, color='red', linestyle=':', alpha=0.7, 
                label=f'-{ci_multiplier:.1f}σ = {-ci_multiplier*std_residual:.3f}')
    
    # Create title with date range if provided
    title = "Residuals Histogram (Kalman Filter vs Raw Data) [lbs]"
    if start_date and end_date:
        title += f"\nDate Range: {start_date} to {end_date}"
    elif start_date:
        title += f"\nFrom: {start_date}"
    elif end_date:
        title += f"\nUntil: {end_date}"
    
    plt.title(title)
    plt.xlabel("Residual (Raw Weight - Kalman Weight) [lbs]")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Add statistics text box
    stats_text = f"""Statistics:
Mean: {mean_residual:.4f} lbs
Std Dev: {std_residual:.4f} lbs
Skewness: {skewness:.4f}
Excess Kurtosis: {excess_kurtosis:.4f}

Normality Tests:
Shapiro-Wilk: p = {shapiro_p:.4f}
Kolmogorov-Smirnov: p = {ks_p:.4f}

Interpretation:
p > 0.05: Residuals appear normal
p ≤ 0.05: Residuals may not be normal
Skewness: 0 = symmetric, >0 = right tail, <0 = left tail
Kurtosis: 0 = normal, >0 = heavy tails, <0 = light tails"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='left', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if no_display:
        plt.close()
    else:
        plt.show()
    
    # Print summary to console
    print(f"\n=== Residuals Analysis ===")
    print(f"Number of residuals: {len(residuals_array)}")
    print(f"Mean residual: {mean_residual:.4f} lbs")
    print(f"Standard deviation: {std_residual:.4f} lbs")
    print(f"Skewness: {skewness:.4f}")
    print(f"Excess kurtosis: {excess_kurtosis:.4f}")
    print(f"Shapiro-Wilk normality test: p = {shapiro_p:.4f}")
    print(f"Kolmogorov-Smirnov normality test: p = {ks_p:.4f}")
    
    if shapiro_p > 0.05:
        print("✓ Residuals appear to be normally distributed (p > 0.05)")
    else:
        print("⚠ Residuals may not be normally distributed (p ≤ 0.05)")
    
    # Additional interpretation based on skewness and kurtosis
    if abs(skewness) > 0.5:
        skew_direction = "right" if skewness > 0 else "left"
        print(f"⚠ Distribution is skewed {skew_direction} (|skewness| = {abs(skewness):.3f} > 0.5)")
    
    if abs(excess_kurtosis) > 0.5:
        kurt_type = "heavy" if excess_kurtosis > 0 else "light"
        print(f"⚠ Distribution has {kurt_type} tails (|excess kurtosis| = {abs(excess_kurtosis):.3f} > 0.5)")
    
    return mean_residual, std_residual, shapiro_p
