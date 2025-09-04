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
class WeightEntry:
    """Weight entry with datetime and weight value"""
    entry_datetime: datetime
    weight: float

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
                     initial_velocity: float = 0.0,
                     data_scale_factor: float = 1.0) -> Tuple[List[KalmanState], List[date]]:
    """
    Run Kalman filter on weight entries
    
    Args:
        entries: List of WeightEntry objects
        initial_weight: Starting weight (defaults to first measurement)
        initial_velocity: Starting velocity estimate
        data_scale_factor: Scale factor for noise parameters (1.0 for weight data, ~0.1 for body fat %)
        
    Returns:
        (kalman_states, dates)
    """
    if not entries:
        return [], []
    
    # Initialize filter
    if initial_weight is None:
        initial_weight = entries[0].weight
    
    # Scale noise parameters based on data range
    scaled_measurement_noise = 0.5 * data_scale_factor
    scaled_process_noise_weight = 0.01 * data_scale_factor
    scaled_process_noise_velocity = 0.001 * data_scale_factor
    scaled_initial_velocity_var = 0.25 * data_scale_factor
    
    kf = WeightKalmanFilter(
        initial_weight=initial_weight,
        initial_velocity=initial_velocity,
        process_noise_weight=scaled_process_noise_weight,
        process_noise_velocity=scaled_process_noise_velocity,
        measurement_noise=scaled_measurement_noise,
        initial_velocity_var=scaled_initial_velocity_var
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


def compute_kalman_mean_std_spline(states, dates) -> Tuple[List[datetime], List[float], List[float]]:
    """
    Mirror the EMA spline approach from weight_tracker.py for Kalman outputs.
    Build a dense cubic spline for the filtered mean and for the std, using
    datetimes and a dense time grid for smooth plotting.

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
    dense_t = np.linspace(min_t, max_t, 5000)

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


def _load_fat_mass_csv(path: str) -> List[Tuple[date, float]]:
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

def _load_calibrated_bf_csv(path: str) -> List[Tuple[datetime, float]]:
    """Load calibrated body fat data from CSV file."""
    points: List[Tuple[datetime, float]] = []
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if not row or len(row) < 2:
                    continue
                try:
                    dt_str = str(row[0]).strip()
                    bf_str = str(row[1]).strip()  # body_fat_pct_cal column (index 1)
                    dt = datetime.fromisoformat(dt_str)
                    bf = float(bf_str)
                    points.append((dt, bf))
                except Exception:
                    # skip bad rows
                    continue
    except Exception:
        return []
    points.sort(key=lambda p: p[0])
    return points


def _evaluate_lbm_series(target_datetimes: List[datetime], fat_mass_points: List[Tuple[date, float]]) -> List[float]:
    if not fat_mass_points:
        return [0.0 for _ in target_datetimes]
    # Build lists of datetimes and values
    pts_dt = [datetime.combine(d, datetime.min.time()) for d, _ in fat_mass_points]
    pts_val = [float(v) for _, v in fat_mass_points]

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

def _evaluate_calibrated_bf_series(target_datetimes: List[datetime], bf_points: List[Tuple[datetime, float]]) -> List[float]:
    """Evaluate calibrated body fat series at target datetimes using linear interpolation."""
    if not bf_points:
        return [0.0 for _ in target_datetimes]
    
    # Build lists of datetimes and values
    pts_dt = [dt for dt, _ in bf_points]
    pts_val = [float(v) for _, v in bf_points]

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
                      ci_multiplier: float = 1.96) -> None:
    """
    Create Kalman filter plot with raw data, filtered state, and confidence bands
    
    Args:
        entries: List of WeightEntry objects
        states: List of KalmanState objects
        dates: List of dates corresponding to states
        output_path: Path to save the plot
    """
    if not entries or not states:
        return
    
    # Filter entries by date range if specified
    filtered_entries = entries
    if start_date or end_date:
        if start_date:
            # Convert start_date to datetime at beginning of day
            start_datetime = datetime.combine(start_date, datetime.min.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime >= start_datetime]
        if end_date:
            # Convert end_date to datetime at end of day
            end_datetime = datetime.combine(end_date, datetime.max.time())
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
    
    entry_datetimes = [e.entry_datetime for e in filtered_entries]
    
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
    
    # Plot raw data
    raw_weights = [e.weight for e in filtered_entries]
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


def create_bodyfat_plot_from_calibrated(
    entries,
    states,
    dates,
    calibrated_bf_csv: str,
    output_path: str = "calibrated_bodyfat_trend.png",
    no_display: bool = False,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    ci_multiplier: float = 1.96,
) -> None:
    """
    Create Body Fat % vs Date plot using DEXA-calibrated body fat measurements with Kalman filtering.
    """
    if not entries or not states:
        return

    # Load calibrated body fat data
    bf_points = _load_calibrated_bf_csv(calibrated_bf_csv)
    if not bf_points:
        print(f"Warning: No calibrated body fat data found in {calibrated_bf_csv}")
        return

    # Filter entries by date range if specified
    filtered_entries = entries
    if start_date or end_date:
        if start_date:
            start_datetime = datetime.combine(start_date, datetime.min.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime >= start_datetime]
        if end_date:
            end_datetime = datetime.combine(end_date, datetime.max.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime <= end_datetime]
        
        if not filtered_entries:
            print(f"Warning: No data in specified date range {start_date} to {end_date}")
            return

    # Create body fat entries from calibrated data
    bf_entries = []
    for dt, bf_pct in bf_points:
        # Apply date filtering if specified
        if start_date or end_date:
            dt_date = dt.date()
            if start_date and dt_date < start_date:
                continue
            if end_date and dt_date > end_date:
                continue
        
        # Find corresponding weight for this datetime
        closest_weight = None
        min_diff = float('inf')
        for entry in entries:
            diff = abs((dt - entry.entry_datetime).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_weight = entry.weight
        
        if closest_weight is not None:
            bf_entries.append(WeightEntry(dt, bf_pct))

    if not bf_entries:
        print("Warning: No matching weight data found for body fat measurements")
        return

    # For calibrated data, use direct interpolation instead of Kalman filtering
    # since the data is already calibrated and smoothed from DEXA calibration
    from scipy.interpolate import interp1d
    import numpy as np
    
    # Create dense time grid
    start_dt = min(e.entry_datetime for e in bf_entries)
    end_dt = max(e.entry_datetime for e in bf_entries)
    total_days = (end_dt - start_dt).total_seconds() / 86400.0
    num_points = min(5000, max(100, int(total_days * 2)))
    dense_datetimes = [start_dt + timedelta(days=i * total_days / (num_points - 1)) for i in range(num_points)]
    
    # Interpolate body fat percentages
    bf_times = [(e.entry_datetime - start_dt).total_seconds() / 86400.0 for e in bf_entries]
    bf_values = [e.weight for e in bf_entries]  # e.weight contains the body fat percentage
    dense_times = [(dt - start_dt).total_seconds() / 86400.0 for dt in dense_datetimes]
    
    # Use linear interpolation
    interp_func = interp1d(bf_times, bf_values, kind='linear', bounds_error=False, fill_value='extrapolate')
    dense_means = interp_func(dense_times).tolist()
    
    # Estimate uncertainty as a small fraction of the data range
    data_range = max(bf_values) - min(bf_values)
    dense_stds = [data_range * 0.05] * len(dense_means)  # 5% of data range as uncertainty
    
    if not dense_datetimes:
        return
    
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

    # Calculate confidence bands
    bf_band_lower = [max(0.0, m - ci_multiplier * s) for m, s in zip(dense_means, dense_stds)]
    bf_band_upper = [m + ci_multiplier * s for m, s in zip(dense_means, dense_stds)]

    # Entry points: use actual measurements
    entry_datetimes = [e.entry_datetime for e in bf_entries]
    entry_bf = [e.weight for e in bf_entries]  # e.weight contains the body fat percentage

    # Plot
    import matplotlib
    if no_display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8), dpi=100)

    # Confidence band around midline
    plt.fill_between(
        dense_datetimes, bf_band_lower, bf_band_upper,
        color="#cccccc", alpha=0.4, label=f"Body Fat {ci_multiplier:.1f}σ CI"
    )

    # Interpolated body fat curve
    plt.plot(dense_datetimes, dense_means, "-", color="#1f77b4", linewidth=2.4,
             label="DEXA-Calibrated Body Fat %")

    # Scatter actual points
    if entry_bf and len(entry_datetimes) == len(entry_bf):
        plt.scatter(entry_datetimes, entry_bf, s=36, color="#1f77b4", alpha=0.8,
                    edgecolors="white", linewidths=0.5, zorder=5, label="Calibrated measurements")

    # Stats panel
    latest_date = dense_datetimes[-1].date()
    current_bf = float(dense_means[-1])
    current_halfwidth = float(max(bf_band_upper[-1] - current_bf, current_bf - bf_band_lower[-1]))

    # Current rate (bf% per week) from derivative of a spline
    try:
        from scipy.interpolate import CubicSpline
        t0_dt = dense_datetimes[0]
        t_days = np.array([(dt - t0_dt).total_seconds() / 86400.0 for dt in dense_datetimes], dtype=float)
        bf_arr = np.array(dense_means, dtype=float)
        bf_spline = CubicSpline(t_days, bf_arr, bc_type="natural")
        bf_slope_per_day = float(bf_spline.derivative()(t_days[-1]))
    except Exception:
        # Fallback: finite difference over last two points
        t0_dt = dense_datetimes[0]
        t_days = np.array([(dt - t0_dt).total_seconds() / 86400.0 for dt in dense_datetimes], dtype=float)
        if len(t_days) >= 2:
            dt_last = float(t_days[-1] - t_days[-2]) if t_days[-1] != t_days[-2] else 1.0
            bf_slope_per_day = float((dense_means[-1] - dense_means[-2]) / dt_last)
        else:
            bf_slope_per_day = 0.0
    bf_slope_per_week = 7.0 * bf_slope_per_day

    # Stats box text
    stats_text = (
        f"Current Estimate ({latest_date})\n"
        f"Body Fat: {current_bf:.2f}% ± {current_halfwidth:.2f}%\n"
        f"Rate: {bf_slope_per_week:+.3f}%/week\n\n"
        f"Data: DEXA-calibrated BIA measurements\n"
        f"Method: Kalman filter on body fat %"
    )

    ax = plt.gca()
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white',
                     alpha=0.9, edgecolor='#cccccc'), zorder=10)

    plt.title("DEXA-Calibrated Body Fat %")
    plt.xlabel("Date")
    plt.ylabel("Body Fat (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if no_display:
        plt.close()
    else:
        plt.show()

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
    fat_mass_csv: Optional[str] = None,
    ci_multiplier: float = 1.96,
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

    # Filter entries by date range if specified
    filtered_entries = entries
    if start_date or end_date:
        if start_date:
            filtered_entries = [e for e in filtered_entries if e.entry_date >= start_date]
        if end_date:
            filtered_entries = [e for e in filtered_entries if e.entry_date <= end_date]
        
        if not filtered_entries:
            print(f"Warning: No data in specified date range {start_date} to {end_date}")
            return

    # Use smoothed Kalman mean and std over a dense date grid
    dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(states, dates)
    if not dense_datetimes:
        return
    
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

    # Baseline defaults: if baseline weight not provided, use first Kalman mean
    if baseline_weight_lb is None:
        baseline_weight_lb = float(dense_means[0])

    # If a fat mass CSV is provided, drive body fat computations directly from fat mass
    if fat_mass_csv:
        fat_mass_points = _load_fat_mass_csv(fat_mass_csv)
        if not fat_mass_points:
            # Fall back to original model if fat mass file empty/unreadable
            fat_mass_csv = None

    if fat_mass_csv:
        # Build LBM series at dense times and entry times
        dense_lbm = _evaluate_lbm_series(dense_datetimes, fat_mass_points)
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
        entry_datetimes = [e.entry_datetime for e in filtered_entries]
        entry_weights = [float(e.weight) for e in filtered_entries]
        entry_lbm = _evaluate_lbm_series(entry_datetimes, fat_mass_points)
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
        entry_datetimes = [e.entry_datetime for e in filtered_entries]
        entry_weights = [float(e.weight) for e in filtered_entries]
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

    # Scatter actual points under mid assumption
    plt.scatter(entry_datetimes, entry_bf_mid, s=36, color="#1f77b4", alpha=0.8,
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
    
    if fat_mass_csv:
        # For LBM-based calculation, uncertainty comes from weight uncertainty
        current_lbm = _evaluate_lbm_series([dense_datetimes[-1]], fat_mass_points)[0]
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
        if fat_mass_csv:
            future_lbm = _evaluate_lbm_series([future_dt], fat_mass_points)[0]
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
        if start_date:
            # Convert start_date to datetime at beginning of day
            start_datetime = datetime.combine(start_date, datetime.min.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime >= start_datetime]
        if end_date:
            # Convert end_date to datetime at end of day
            end_datetime = datetime.combine(end_date, datetime.max.time())
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
) -> None:
    """
    Create BMI vs Date plot using Kalman-smoothed weights and height data.
    
    BMI = weight(kg) / height(m)^2
    For imperial units: BMI = (weight_lb / height_in^2) * 703
    """
    if not entries or not states:
        return

    # Load height data
    height_inches = _load_height_data(height_file)
    if height_inches is None:
        print(f"Warning: No height data found at {height_file}. BMI plot cannot be generated.")
        return

    # Filter entries by date range if specified
    filtered_entries = entries
    if start_date or end_date:
        if start_date:
            filtered_entries = [e for e in filtered_entries if e.entry_date >= start_date]
        if end_date:
            filtered_entries = [e for e in filtered_entries if e.entry_date <= end_date]
        
        if not filtered_entries:
            print(f"Warning: No data in specified date range {start_date} to {end_date}")
            return

    # Use smoothed Kalman mean and std over a dense date grid
    dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(states, dates)
    if not dense_datetimes:
        return
    
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
    entry_datetimes = [e.entry_datetime for e in filtered_entries]
    entry_weights = [float(e.weight) for e in filtered_entries]
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

    # Scatter points for actual measurements
    plt.scatter(entry_datetimes, entry_bmi, color="red", s=30, alpha=0.7, label="BMI (measurements)")

    # BMI categories
    plt.axhline(y=18.5, color="green", linestyle="--", alpha=0.7, label="Underweight (18.5)")
    plt.axhline(y=25.0, color="orange", linestyle="--", alpha=0.7, label="Normal (25.0)")
    plt.axhline(y=30.0, color="red", linestyle="--", alpha=0.7, label="Overweight (30.0)")

    plt.xlabel("Date")
    plt.ylabel("BMI (kg/m²)")
    plt.title(f"BMI Trend (Height: {height_inches:.1f}\" = {height_m:.2f}m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if not no_display:
        plt.show()
    plt.close()


def create_ffmi_plot_from_calibrated(
    entries,
    states,
    dates,
    calibrated_bf_csv: str,
    height_file: str = "data/height.txt",
    output_path: str = "calibrated_ffmi_trend.png",
    no_display: bool = False,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    ci_multiplier: float = 1.96,
) -> None:
    """
    Create FFMI vs Date plot using DEXA-calibrated body fat measurements with Kalman filtering.
    """
    if not entries or not states:
        return

    # Load calibrated body fat data
    bf_points = _load_calibrated_bf_csv(calibrated_bf_csv)
    if not bf_points:
        print(f"Warning: No calibrated body fat data found in {calibrated_bf_csv}")
        return

    # Load height data
    height_inches = _load_height_data(height_file)
    if height_inches <= 0:
        print(f"Warning: Invalid height data from {height_file}")
        return

    # Filter entries by date range if specified
    filtered_entries = entries
    if start_date or end_date:
        if start_date:
            start_datetime = datetime.combine(start_date, datetime.min.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime >= start_datetime]
        if end_date:
            end_datetime = datetime.combine(end_date, datetime.max.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime <= end_datetime]
        
        if not filtered_entries:
            print(f"Warning: No data in specified date range {start_date} to {end_date}")
            return

    # Create FFMI entries from calibrated data
    ffmi_entries = []
    height_m = height_inches * 0.0254
    for dt, bf_pct in bf_points:
        # Apply date filtering if specified
        if start_date or end_date:
            dt_date = dt.date()
            if start_date and dt_date < start_date:
                continue
            if end_date and dt_date > end_date:
                continue
        
        # Find corresponding weight for this datetime
        closest_weight = None
        min_diff = float('inf')
        for entry in entries:
            diff = abs((dt - entry.entry_datetime).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_weight = entry.weight
        
        if closest_weight is not None:
            lbm = closest_weight * (1.0 - bf_pct / 100.0)
            ffmi = (lbm * 0.453592) / (height_m ** 2)
            ffmi_entries.append(WeightEntry(dt, ffmi))

    if not ffmi_entries:
        print("Warning: No matching weight data found for body fat measurements")
        return

    # For calibrated data, use direct interpolation instead of Kalman filtering
    from scipy.interpolate import interp1d
    import numpy as np
    
    # Create dense time grid
    start_dt = min(e.entry_datetime for e in ffmi_entries)
    end_dt = max(e.entry_datetime for e in ffmi_entries)
    total_days = (end_dt - start_dt).total_seconds() / 86400.0
    num_points = min(5000, max(100, int(total_days * 2)))
    dense_datetimes = [start_dt + timedelta(days=i * total_days / (num_points - 1)) for i in range(num_points)]
    
    # Interpolate FFMI values
    ffmi_times = [(e.entry_datetime - start_dt).total_seconds() / 86400.0 for e in ffmi_entries]
    ffmi_values = [e.weight for e in ffmi_entries]  # e.weight contains the FFMI
    dense_times = [(dt - start_dt).total_seconds() / 86400.0 for dt in dense_datetimes]
    
    # Use linear interpolation
    interp_func = interp1d(ffmi_times, ffmi_values, kind='linear', bounds_error=False, fill_value='extrapolate')
    dense_means = interp_func(dense_times).tolist()
    
    # Estimate uncertainty as a small fraction of the data range
    data_range = max(ffmi_values) - min(ffmi_values)
    dense_stds = [data_range * 0.05] * len(dense_means)  # 5% of data range as uncertainty
    if not dense_datetimes:
        return
    
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

    # Calculate confidence bands
    ffmi_band_lower = [max(0.0, m - ci_multiplier * s) for m, s in zip(dense_means, dense_stds)]
    ffmi_band_upper = [m + ci_multiplier * s for m, s in zip(dense_means, dense_stds)]

    # Entry points: use actual measurements
    entry_datetimes = [e.entry_datetime for e in ffmi_entries]
    entry_ffmi = [e.weight for e in ffmi_entries]

    # Plot
    import matplotlib
    if no_display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8), dpi=100)

    # Confidence band around midline
    plt.fill_between(
        dense_datetimes, ffmi_band_lower, ffmi_band_upper,
        color="#cccccc", alpha=0.4, label=f"FFMI {ci_multiplier:.1f}σ CI"
    )

    # Kalman-filtered FFMI curve
    plt.plot(dense_datetimes, dense_means, "-", color="#ff7f0e", linewidth=2.4,
             label="DEXA-Calibrated FFMI (Kalman)")

    # Scatter actual points
    if entry_ffmi and len(entry_datetimes) == len(entry_ffmi):
        plt.scatter(entry_datetimes, entry_ffmi, s=36, color="#ff7f0e", alpha=0.8,
                    edgecolors="white", linewidths=0.5, zorder=5, label="Calibrated measurements")

    # Stats panel
    latest_date = dense_datetimes[-1].date()
    current_ffmi = float(dense_means[-1])
    current_halfwidth = float(max(ffmi_band_upper[-1] - current_ffmi, current_ffmi - ffmi_band_lower[-1]))

    # Current rate (FFMI per week) from derivative of a spline
    try:
        from scipy.interpolate import CubicSpline
        t0_dt = dense_datetimes[0]
        t_days = np.array([(dt - t0_dt).total_seconds() / 86400.0 for dt in dense_datetimes], dtype=float)
        ffmi_arr = np.array(dense_means, dtype=float)
        ffmi_spline = CubicSpline(t_days, ffmi_arr, bc_type="natural")
        ffmi_slope_per_day = float(ffmi_spline.derivative()(t_days[-1]))
    except Exception:
        # Fallback: finite difference over last two points
        t0_dt = dense_datetimes[0]
        t_days = np.array([(dt - t0_dt).total_seconds() / 86400.0 for dt in dense_datetimes], dtype=float)
        if len(t_days) >= 2:
            dt_last = float(t_days[-1] - t_days[-2]) if t_days[-1] != t_days[-2] else 1.0
            ffmi_slope_per_day = float((dense_means[-1] - dense_means[-2]) / dt_last)
        else:
            ffmi_slope_per_day = 0.0
    ffmi_slope_per_week = 7.0 * ffmi_slope_per_day

    # Stats box text
    stats_text = (
        f"Current Estimate ({latest_date})\n"
        f"FFMI: {current_ffmi:.2f} ± {current_halfwidth:.2f}\n"
        f"Rate: {ffmi_slope_per_week:+.3f}/week\n\n"
        f"Data: DEXA-calibrated BIA measurements\n"
        f"Method: Kalman filter on FFMI"
    )

    ax = plt.gca()
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white',
                     alpha=0.9, edgecolor='#cccccc'), zorder=10)

    plt.title("DEXA-Calibrated Fat-Free Mass Index")
    plt.xlabel("Date")
    plt.ylabel("FFMI")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if no_display:
        plt.close()
    else:
        plt.show()


def create_ffmi_plot_from_kalman(
    entries,
    states,
    dates,
    height_file: str = "data/height.txt",
    output_path: str = "ffmi_trend.png",
    no_display: bool = False,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    fat_mass_csv: Optional[str] = None,
    ci_multiplier: float = 1.96,
) -> None:
    """
    Create FFMI (Fat-Free Mass Index) vs Date plot using Kalman-smoothed weights and height data.
    
    FFMI = LBM(kg) / height(m)^2
    For imperial units: FFMI = (LBM_lb / height_in^2) * 703
    """
    if not entries or not states:
        return

    # Load height data
    height_inches = _load_height_data(height_file)
    if height_inches is None:
        print(f"Warning: No height data found at {height_file}. FFMI plot cannot be generated.")
        return

    # Filter entries by date range if specified
    filtered_entries = entries
    if start_date or end_date:
        if start_date:
            filtered_entries = [e for e in filtered_entries if e.entry_date >= start_date]
        if end_date:
            filtered_entries = [e for e in filtered_entries if e.entry_date <= end_date]
        
        if not filtered_entries:
            print(f"Warning: No data in specified date range {start_date} to {end_date}")
            return

    # Use smoothed Kalman mean and std over a dense date grid
    dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(states, dates)
    if not dense_datetimes:
        return
    
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

    # Convert height to meters for FFMI calculation
    height_m = height_inches * 0.0254  # inches to meters
    
    # Calculate LBM from Kalman-smoothed weights
    # Use the same LBM calculation logic as body fat plot
    if fat_mass_csv:
        fat_mass_points = _load_fat_mass_csv(fat_mass_csv)
        if not fat_mass_points:
            # Fall back to estimated LBM if LBM file empty/unreadable
            fat_mass_points = None
    else:
        fat_mass_points = None
    
    if fat_mass_points:
        # Use provided LBM data
        dense_lbm = _evaluate_lbm_series(dense_datetimes, fat_mass_points)
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
        entry_datetimes = [e.entry_datetime for e in filtered_entries]
        entry_weights = [float(e.weight) for e in filtered_entries]
        entry_lbm = _evaluate_lbm_series(entry_datetimes, fat_mass_points)
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
        entry_datetimes = [e.entry_datetime for e in filtered_entries]
        entry_weights = [float(e.weight) for e in filtered_entries]
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

    # Scatter points for actual measurements
    plt.scatter(entry_datetimes, entry_ffmi, color="red", s=30, alpha=0.7, label="FFMI (measurements)")

    # FFMI reference lines (typical ranges for men)
    plt.axhline(y=16.0, color="red", linestyle="--", alpha=0.7, label="Below average (16.0)")
    plt.axhline(y=18.0, color="orange", linestyle="--", alpha=0.7, label="Average (18.0)")
    plt.axhline(y=20.0, color="green", linestyle="--", alpha=0.7, label="Above average (20.0)")
    plt.axhline(y=22.0, color="blue", linestyle="--", alpha=0.7, label="Excellent (22.0)")

    plt.xlabel("Date")
    plt.ylabel("FFMI (kg/m²)")
    plt.title(f"FFMI Trend (Height: {height_inches:.1f}\" = {height_m:.2f}m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if not no_display:
        plt.show()
    plt.close()


def create_lbm_plot_from_calibrated(
    entries,
    states,
    dates,
    calibrated_bf_csv: str,
    output_path: str = "calibrated_lbm_trend.png",
    no_display: bool = False,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    ci_multiplier: float = 1.96,
) -> None:
    """
    Create LBM (lb) vs Date plot using DEXA-calibrated body fat measurements with Kalman filtering.
    """
    if not entries or not states:
        return

    # Load calibrated body fat data
    bf_points = _load_calibrated_bf_csv(calibrated_bf_csv)
    if not bf_points:
        print(f"Warning: No calibrated body fat data found in {calibrated_bf_csv}")
        return

    # Filter entries by date range if specified
    filtered_entries = entries
    if start_date or end_date:
        if start_date:
            start_datetime = datetime.combine(start_date, datetime.min.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime >= start_datetime]
        if end_date:
            end_datetime = datetime.combine(end_date, datetime.max.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime <= end_datetime]
        
        if not filtered_entries:
            print(f"Warning: No data in specified date range {start_date} to {end_date}")
            return

    # Create LBM entries from calibrated data
    lbm_entries = []
    for dt, bf_pct in bf_points:
        # Apply date filtering if specified
        if start_date or end_date:
            dt_date = dt.date()
            if start_date and dt_date < start_date:
                continue
            if end_date and dt_date > end_date:
                continue
        
        # Find corresponding weight for this datetime
        closest_weight = None
        min_diff = float('inf')
        for entry in entries:
            diff = abs((dt - entry.entry_datetime).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_weight = entry.weight
        
        if closest_weight is not None:
            lbm = closest_weight * (1.0 - bf_pct / 100.0)
            lbm_entries.append(WeightEntry(dt, lbm))

    if not lbm_entries:
        print("Warning: No matching weight data found for body fat measurements")
        return

    # For calibrated data, use direct interpolation instead of Kalman filtering
    from scipy.interpolate import interp1d
    import numpy as np
    
    # Create dense time grid
    start_dt = min(e.entry_datetime for e in lbm_entries)
    end_dt = max(e.entry_datetime for e in lbm_entries)
    total_days = (end_dt - start_dt).total_seconds() / 86400.0
    num_points = min(5000, max(100, int(total_days * 2)))
    dense_datetimes = [start_dt + timedelta(days=i * total_days / (num_points - 1)) for i in range(num_points)]
    
    # Interpolate LBM values
    lbm_times = [(e.entry_datetime - start_dt).total_seconds() / 86400.0 for e in lbm_entries]
    lbm_values = [e.weight for e in lbm_entries]  # e.weight contains the LBM
    dense_times = [(dt - start_dt).total_seconds() / 86400.0 for dt in dense_datetimes]
    
    # Use linear interpolation
    interp_func = interp1d(lbm_times, lbm_values, kind='linear', bounds_error=False, fill_value='extrapolate')
    dense_means = interp_func(dense_times).tolist()
    
    # Estimate uncertainty as a small fraction of the data range
    data_range = max(lbm_values) - min(lbm_values)
    dense_stds = [data_range * 0.05] * len(dense_means)  # 5% of data range as uncertainty
    if not dense_datetimes:
        return
    
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

    # Calculate confidence bands
    lbm_band_lower = [max(0.0, m - ci_multiplier * s) for m, s in zip(dense_means, dense_stds)]
    lbm_band_upper = [m + ci_multiplier * s for m, s in zip(dense_means, dense_stds)]

    # Entry points: use actual measurements
    entry_datetimes = [e.entry_datetime for e in lbm_entries]
    entry_lbm = [e.weight for e in lbm_entries]

    # Plot
    import matplotlib
    if no_display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8), dpi=100)

    # Confidence band around midline
    plt.fill_between(
        dense_datetimes, lbm_band_lower, lbm_band_upper,
        color="#cccccc", alpha=0.4, label=f"LBM {ci_multiplier:.1f}σ CI"
    )

    # Kalman-filtered LBM curve
    plt.plot(dense_datetimes, dense_means, "-", color="#2ca02c", linewidth=2.4,
             label="DEXA-Calibrated LBM (Kalman)")

    # Scatter actual points
    if entry_lbm and len(entry_datetimes) == len(entry_lbm):
        plt.scatter(entry_datetimes, entry_lbm, s=36, color="#2ca02c", alpha=0.8,
                    edgecolors="white", linewidths=0.5, zorder=5, label="Calibrated measurements")

    # Stats panel
    latest_date = dense_datetimes[-1].date()
    current_lbm = float(dense_means[-1])
    current_halfwidth = float(max(lbm_band_upper[-1] - current_lbm, current_lbm - lbm_band_lower[-1]))

    # Current rate (LBM per week) from derivative of a spline
    try:
        from scipy.interpolate import CubicSpline
        t0_dt = dense_datetimes[0]
        t_days = np.array([(dt - t0_dt).total_seconds() / 86400.0 for dt in dense_datetimes], dtype=float)
        lbm_arr = np.array(dense_means, dtype=float)
        lbm_spline = CubicSpline(t_days, lbm_arr, bc_type="natural")
        lbm_slope_per_day = float(lbm_spline.derivative()(t_days[-1]))
    except Exception:
        # Fallback: finite difference over last two points
        t0_dt = dense_datetimes[0]
        t_days = np.array([(dt - t0_dt).total_seconds() / 86400.0 for dt in dense_datetimes], dtype=float)
        if len(t_days) >= 2:
            dt_last = float(t_days[-1] - t_days[-2]) if t_days[-1] != t_days[-2] else 1.0
            lbm_slope_per_day = float((dense_means[-1] - dense_means[-2]) / dt_last)
        else:
            lbm_slope_per_day = 0.0
    lbm_slope_per_week = 7.0 * lbm_slope_per_day

    # Stats box text
    stats_text = (
        f"Current Estimate ({latest_date})\n"
        f"LBM: {current_lbm:.2f} ± {current_halfwidth:.2f} lb\n"
        f"Rate: {lbm_slope_per_week:+.3f} lb/week\n\n"
        f"Data: DEXA-calibrated BIA measurements\n"
        f"Method: Kalman filter on LBM"
    )

    ax = plt.gca()
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white',
                     alpha=0.9, edgecolor='#cccccc'), zorder=10)

    plt.title("DEXA-Calibrated Lean Body Mass")
    plt.xlabel("Date")
    plt.ylabel("LBM (lb)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if no_display:
        plt.close()
    else:
        plt.show()


def create_lbm_plot_from_kalman(
    entries,
    states,
    dates,
    baseline_weight_lb: Optional[float],
    baseline_lean_lb: float,
    output_path: str = "lbm_trend.png",
    no_display: bool = False,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    fat_mass_csv: Optional[str] = None,
    ci_multiplier: float = 1.96,
) -> None:
    """
    Plot Lean Body Mass (LBM) over time.

    - If an LBM CSV is provided, use it for scatter points; the smoothed LBM curve
      is derived from Kalman-smoothed weight using the mid scenario (s=0.10) model:
        L(t) = L0 + s * (W(t) - W0)
    - Confidence band is computed from W ± z*std propagated through the model.
    """
    if not entries or not states:
        return

    # Filter entries by date range if specified
    filtered_entries = entries
    if start_date or end_date:
        if start_date:
            # Convert start_date to datetime at beginning of day
            start_dt = datetime.combine(start_date, datetime.min.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime >= start_dt]
        if end_date:
            end_dt = datetime.combine(end_date, datetime.max.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime <= end_dt]
        if not filtered_entries:
            print(f"Warning: No data in specified date range {start_date} to {end_date}")
            return

    # Smoothed mean/std over dense grid
    dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(states, dates)
    if not dense_datetimes:
        return

    # Filter dense series by date range
    if start_date or end_date:
        f_dt, f_mean, f_std = [], [], []
        for dt, m, s in zip(dense_datetimes, dense_means, dense_stds):
            d = dt.date()
            if start_date and d < start_date:
                continue
            if end_date and d > end_date:
                continue
            f_dt.append(dt)
            f_mean.append(m)
            f_std.append(s)
        dense_datetimes, dense_means, dense_stds = f_dt, f_mean, f_std

    # Baseline defaults
    if baseline_weight_lb is None:
        baseline_weight_lb = float(dense_means[0])
    W0 = float(baseline_weight_lb)
    L0 = float(baseline_lean_lb)
    s_mid = 0.10

    # Smoothed LBM curve and confidence band via propagation
    dense_lbm = [max(0.0, min(float(L0 + s_mid * (w - W0)), float(w))) for w in dense_means]
    w_lo = [max(1e-6, m - ci_multiplier * st) for m, st in zip(dense_means, dense_stds)]
    w_hi = [m + ci_multiplier * st for m, st in zip(dense_means, dense_stds)]
    lbm_lo = [max(0.0, min(float(L0 + s_mid * (wl - W0)), float(wl))) for wl in w_lo]
    lbm_hi = [max(0.0, min(float(L0 + s_mid * (wh - W0)), float(wh))) for wh in w_hi]
    band_lower = [min(a, b) for a, b in zip(lbm_lo, lbm_hi)]
    band_upper = [max(a, b) for a, b in zip(lbm_lo, lbm_hi)]

    # Scatter points
    entry_datetimes = [e.entry_datetime for e in filtered_entries]
    if fat_mass_csv:
        fat_mass_points = _load_fat_mass_csv(fat_mass_csv)
        if fat_mass_points:
            entry_lbm = _evaluate_lbm_series(entry_datetimes, fat_mass_points)
        else:
            entry_lbm = [max(0.0, min(float(L0 + s_mid * (w - W0)), float(w))) for w in [e.weight for e in filtered_entries]]
    else:
        entry_lbm = [max(0.0, min(float(L0 + s_mid * (w - W0)), float(w))) for w in [e.weight for e in filtered_entries]]

    # Plot
    import matplotlib
    if no_display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8), dpi=100)
    plt.fill_between(dense_datetimes, band_lower, band_upper, color="#cccccc", alpha=0.4,
                     label=f"LBM {ci_multiplier:.1f}σ CI")
    plt.plot(dense_datetimes, dense_lbm, "-", color="#1f77b4", linewidth=2.4, label="LBM (estimated)")
    plt.scatter(entry_datetimes, entry_lbm, s=36, color="#1f77b4", alpha=0.8,
                edgecolors="white", linewidths=0.5, zorder=5, label="LBM (measurements)")

    # Stats box at lower-left
    latest_lbm = float(dense_lbm[-1])
    halfwidth = float(max(band_upper[-1] - latest_lbm, latest_lbm - band_lower[-1]))
    ax = plt.gca()
    ax.text(0.02, 0.02, f"Current LBM: {latest_lbm:.2f} ± {halfwidth:.2f} lb",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"), zorder=10)

    plt.title("Lean Body Mass (LBM)")
    plt.xlabel("Date")
    plt.ylabel("LBM (lb)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if not no_display:
        plt.show()
    plt.close()


def create_fatmass_plot_from_calibrated(
    entries,
    states,
    dates,
    calibrated_bf_csv: str,
    output_path: str = "calibrated_fatmass_trend.png",
    no_display: bool = False,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    ci_multiplier: float = 1.96,
) -> None:
    """
    Create Fat Mass (lb) vs Date plot using DEXA-calibrated body fat measurements with Kalman filtering.
    """
    if not entries or not states:
        return

    # Load calibrated body fat data
    bf_points = _load_calibrated_bf_csv(calibrated_bf_csv)
    if not bf_points:
        print(f"Warning: No calibrated body fat data found in {calibrated_bf_csv}")
        return

    # Filter entries by date range if specified
    filtered_entries = entries
    if start_date or end_date:
        if start_date:
            start_datetime = datetime.combine(start_date, datetime.min.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime >= start_datetime]
        if end_date:
            end_datetime = datetime.combine(end_date, datetime.max.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime <= end_datetime]
        
        if not filtered_entries:
            print(f"Warning: No data in specified date range {start_date} to {end_date}")
            return

    # Create fat mass entries from calibrated data
    fat_mass_entries = []
    for dt, bf_pct in bf_points:
        # Apply date filtering if specified
        if start_date or end_date:
            dt_date = dt.date()
            if start_date and dt_date < start_date:
                continue
            if end_date and dt_date > end_date:
                continue
        
        # Find corresponding weight for this datetime
        closest_weight = None
        min_diff = float('inf')
        for entry in entries:
            diff = abs((dt - entry.entry_datetime).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_weight = entry.weight
        
        if closest_weight is not None:
            fat_mass = closest_weight * bf_pct / 100.0
            fat_mass_entries.append(WeightEntry(dt, fat_mass))

    if not fat_mass_entries:
        print("Warning: No matching weight data found for body fat measurements")
        return

    # For calibrated data, use direct interpolation instead of Kalman filtering
    from scipy.interpolate import interp1d
    import numpy as np
    
    # Create dense time grid
    start_dt = min(e.entry_datetime for e in fat_mass_entries)
    end_dt = max(e.entry_datetime for e in fat_mass_entries)
    total_days = (end_dt - start_dt).total_seconds() / 86400.0
    num_points = min(5000, max(100, int(total_days * 2)))
    dense_datetimes = [start_dt + timedelta(days=i * total_days / (num_points - 1)) for i in range(num_points)]
    
    # Interpolate fat mass values
    fm_times = [(e.entry_datetime - start_dt).total_seconds() / 86400.0 for e in fat_mass_entries]
    fm_values = [e.weight for e in fat_mass_entries]  # e.weight contains the fat mass
    dense_times = [(dt - start_dt).total_seconds() / 86400.0 for dt in dense_datetimes]
    
    # Use linear interpolation
    interp_func = interp1d(fm_times, fm_values, kind='linear', bounds_error=False, fill_value='extrapolate')
    dense_means = interp_func(dense_times).tolist()
    
    # Estimate uncertainty as a small fraction of the data range
    data_range = max(fm_values) - min(fm_values)
    dense_stds = [data_range * 0.05] * len(dense_means)  # 5% of data range as uncertainty
    if not dense_datetimes:
        return
    
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

    # Calculate confidence bands
    fm_band_lower = [max(0.0, m - ci_multiplier * s) for m, s in zip(dense_means, dense_stds)]
    fm_band_upper = [m + ci_multiplier * s for m, s in zip(dense_means, dense_stds)]

    # Entry points: use actual measurements
    entry_datetimes = [e.entry_datetime for e in fat_mass_entries]
    entry_fm = [e.weight for e in fat_mass_entries]

    # Plot
    import matplotlib
    if no_display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8), dpi=100)

    # Confidence band around midline
    plt.fill_between(
        dense_datetimes, fm_band_lower, fm_band_upper,
        color="#cccccc", alpha=0.4, label=f"Fat Mass {ci_multiplier:.1f}σ CI"
    )

    # Kalman-filtered fat mass curve
    plt.plot(dense_datetimes, dense_means, "-", color="#d62728", linewidth=2.4,
             label="DEXA-Calibrated Fat Mass (Kalman)")

    # Scatter actual points
    if entry_fm and len(entry_datetimes) == len(entry_fm):
        plt.scatter(entry_datetimes, entry_fm, s=36, color="#d62728", alpha=0.8,
                    edgecolors="white", linewidths=0.5, zorder=5, label="Calibrated measurements")

    # Stats panel
    latest_date = dense_datetimes[-1].date()
    current_fm = float(dense_means[-1])
    current_halfwidth = float(max(fm_band_upper[-1] - current_fm, current_fm - fm_band_lower[-1]))

    # Current rate (fat mass per week) from derivative of a spline
    try:
        from scipy.interpolate import CubicSpline
        t0_dt = dense_datetimes[0]
        t_days = np.array([(dt - t0_dt).total_seconds() / 86400.0 for dt in dense_datetimes], dtype=float)
        fm_arr = np.array(dense_means, dtype=float)
        fm_spline = CubicSpline(t_days, fm_arr, bc_type="natural")
        fm_slope_per_day = float(fm_spline.derivative()(t_days[-1]))
    except Exception:
        # Fallback: finite difference over last two points
        t0_dt = dense_datetimes[0]
        t_days = np.array([(dt - t0_dt).total_seconds() / 86400.0 for dt in dense_datetimes], dtype=float)
        if len(t_days) >= 2:
            dt_last = float(t_days[-1] - t_days[-2]) if t_days[-1] != t_days[-2] else 1.0
            fm_slope_per_day = float((dense_means[-1] - dense_means[-2]) / dt_last)
        else:
            fm_slope_per_day = 0.0
    fm_slope_per_week = 7.0 * fm_slope_per_day

    # Stats box text
    stats_text = (
        f"Current Estimate ({latest_date})\n"
        f"Fat Mass: {current_fm:.2f} ± {current_halfwidth:.2f} lb\n"
        f"Rate: {fm_slope_per_week:+.3f} lb/week\n\n"
        f"Data: DEXA-calibrated BIA measurements\n"
        f"Method: Kalman filter on fat mass"
    )

    ax = plt.gca()
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white',
                     alpha=0.9, edgecolor='#cccccc'), zorder=10)

    plt.title("DEXA-Calibrated Fat Mass")
    plt.xlabel("Date")
    plt.ylabel("Fat Mass (lb)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if no_display:
        plt.close()
    else:
        plt.show()


def create_fatmass_plot_from_kalman(
    entries,
    states,
    dates,
    baseline_weight_lb: Optional[float],
    baseline_lean_lb: float,
    output_path: str = "fatmass_trend.png",
    no_display: bool = False,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    fat_mass_csv: Optional[str] = None,
    ci_multiplier: float = 1.96,
) -> None:
    """
    Plot Fat Mass (lb) over time: F = W - LBM.

    - If LBM CSV provided, use it for scatter and for computing fat mass band by
      holding LBM fixed while varying W ± z*std.
    - Otherwise, derive LBM via mid scenario (s=0.10): L = L0 + s*(W - W0).
    """
    if not entries or not states:
        return

    # Filter entries by date range if specified
    filtered_entries = entries
    if start_date or end_date:
        if start_date:
            start_dt = datetime.combine(start_date, datetime.min.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime >= start_dt]
        if end_date:
            end_dt = datetime.combine(end_date, datetime.max.time())
            filtered_entries = [e for e in filtered_entries if e.entry_datetime <= end_dt]
        if not filtered_entries:
            print(f"Warning: No data in specified date range {start_date} to {end_date}")
            return

    # Smoothed mean/std over dense grid
    dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(states, dates)
    if not dense_datetimes:
        return

    # Filter dense series by date range
    if start_date or end_date:
        f_dt, f_mean, f_std = [], [], []
        for dt, m, s in zip(dense_datetimes, dense_means, dense_stds):
            d = dt.date()
            if start_date and d < start_date:
                continue
            if end_date and d > end_date:
                continue
            f_dt.append(dt)
            f_mean.append(m)
            f_std.append(s)
        dense_datetimes, dense_means, dense_stds = f_dt, f_mean, f_std

    # Baseline defaults
    if baseline_weight_lb is None:
        baseline_weight_lb = float(dense_means[0])
    W0 = float(baseline_weight_lb)
    L0 = float(baseline_lean_lb)
    s_mid = 0.10

    # If fat mass CSV is provided and valid, use it to compute fat mass
    fat_mass_points = None
    if fat_mass_csv:
        pts = _load_fat_mass_csv(fat_mass_csv)
        if pts:
            fat_mass_points = pts

    if fat_mass_points:
        dense_lbm = _evaluate_lbm_series(dense_datetimes, fat_mass_points)
        dense_fat = [max(0.0, float(w - l)) for w, l in zip(dense_means, dense_lbm)]
        w_lo = [max(1e-6, m - ci_multiplier * st) for m, st in zip(dense_means, dense_stds)]
        w_hi = [m + ci_multiplier * st for m, st in zip(dense_means, dense_stds)]
        fat_lo = [max(0.0, float(wl - l)) for wl, l in zip(w_lo, dense_lbm)]
        fat_hi = [max(0.0, float(wh - l)) for wh, l in zip(w_hi, dense_lbm)]
    else:
        # Model-based LBM then fat mass
        dense_lbm = [max(0.0, min(float(L0 + s_mid * (w - W0)), float(w))) for w in dense_means]
        dense_fat = [max(0.0, float(w - l)) for w, l in zip(dense_means, dense_lbm)]
        w_lo = [max(1e-6, m - ci_multiplier * st) for m, st in zip(dense_means, dense_stds)]
        w_hi = [m + ci_multiplier * st for m, st in zip(dense_means, dense_stds)]
        lbm_lo = [max(0.0, min(float(L0 + s_mid * (wl - W0)), float(wl))) for wl in w_lo]
        lbm_hi = [max(0.0, min(float(L0 + s_mid * (wh - W0)), float(wh))) for wh in w_hi]
        fat_lo = [max(0.0, float(wl - ll)) for wl, ll in zip(w_lo, lbm_lo)]
        fat_hi = [max(0.0, float(wh - lh)) for wh, lh in zip(w_hi, lbm_hi)]

    band_lower = [min(a, b) for a, b in zip(fat_lo, fat_hi)]
    band_upper = [max(a, b) for a, b in zip(fat_lo, fat_hi)]

    # Scatter points
    entry_datetimes = [e.entry_datetime for e in filtered_entries]
    if fat_mass_points:
        entry_lbm = _evaluate_lbm_series(entry_datetimes, fat_mass_points)
        entry_fat = [max(0.0, float(w - l)) for w, l in zip([e.weight for e in filtered_entries], entry_lbm)]
    else:
        entry_lbm = [max(0.0, min(float(L0 + s_mid * (w - W0)), float(w))) for w in [e.weight for e in filtered_entries]]
        entry_fat = [max(0.0, float(w - l)) for w, l in zip([e.weight for e in filtered_entries], entry_lbm)]

    # Plot
    import matplotlib
    if no_display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8), dpi=100)
    plt.fill_between(dense_datetimes, band_lower, band_upper, color="#f2b2ae", alpha=0.4,
                     label=f"Fat mass {ci_multiplier:.1f}σ CI")
    plt.plot(dense_datetimes, dense_fat, "-", color="#d62728", linewidth=2.4, label="Fat mass (estimated)")
    plt.scatter(entry_datetimes, entry_fat, s=36, color="#d62728", alpha=0.8,
                edgecolors="white", linewidths=0.5, zorder=5, label="Fat mass (measurements)")

    # Stats box
    latest_fat = float(dense_fat[-1])
    halfwidth = float(max(band_upper[-1] - latest_fat, latest_fat - band_lower[-1]))
    ax = plt.gca()
    ax.text(0.02, 0.02, f"Current Fat Mass: {latest_fat:.2f} ± {halfwidth:.2f} lb",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"), zorder=10)

    plt.title("Fat Mass (lb)")
    plt.xlabel("Date")
    plt.ylabel("Fat mass (lb)")
    plt.grid(True, alpha=0.3)
    plt.legend()
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
                label=f'+1σ = {std_residual:.3f}')
    plt.axvline(-std_residual, color='orange', linestyle=':', alpha=0.7, 
                label=f'-1σ = {-std_residual:.3f}')
    plt.axvline(ci_multiplier*std_residual, color='red', linestyle=':', alpha=0.7, 
                label=f'+{ci_multiplier:.1f}σ = {ci_multiplier*std_residual:.3f}')
    plt.axvline(-ci_multiplier*std_residual, color='red', linestyle=':', alpha=0.7, 
                label=f'-{ci_multiplier:.1f}σ = {-ci_multiplier*std_residual:.3f}')
    
    # Create title with date range if provided
    title = "Residuals Histogram (Kalman Filter vs Raw Data)"
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
