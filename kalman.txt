#!/usr/bin/env python3

import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


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
                 measurement_noise: float = 0.5):
        """
        Initialize Kalman filter
        
        Args:
            initial_weight: Starting weight estimate
            initial_velocity: Starting velocity estimate (lbs/day)
            process_noise_weight: Process noise for weight (variance per day)
            process_noise_velocity: Process noise for velocity (variance per day)
            measurement_noise: Measurement noise variance
        """
        self.measurement_noise = measurement_noise
        
        # Initial state
        self.state = KalmanState(
            weight=initial_weight,
            velocity=initial_velocity,
            weight_var=1.0,  # Initial uncertainty
            velocity_var=0.1,  # Initial velocity uncertainty
            weight_velocity_cov=0.0
        )
        
        # Process noise matrix (per day)
        self.Q = np.array([
            [process_noise_weight, 0.0],
            [0.0, process_noise_velocity]
        ])
        
        # Measurement matrix (we only measure weight)
        self.H = np.array([[1.0, 0.0]])
        
        # Measurement noise
        self.R = np.array([[measurement_noise]])
    
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
        ])
        
        # Predict state
        new_weight = self.state.weight + self.state.velocity * dt_days
        new_velocity = self.state.velocity
        
        # Predict covariance
        F_T = F.T
        new_weight_var = (F[0, 0]**2 * self.state.weight_var + 
                         F[0, 1]**2 * self.state.velocity_var + 
                         2 * F[0, 0] * F[0, 1] * self.state.weight_velocity_cov)
        new_velocity_var = (F[1, 0]**2 * self.state.weight_var + 
                           F[1, 1]**2 * self.state.velocity_var + 
                           2 * F[1, 0] * F[1, 1] * self.state.weight_velocity_cov)
        new_cov = (F[0, 0] * F[1, 0] * self.state.weight_var + 
                   F[0, 1] * F[1, 1] * self.state.velocity_var + 
                   F[0, 0] * F[1, 1] * self.state.weight_velocity_cov + 
                   F[0, 1] * F[1, 0] * self.state.weight_velocity_cov)
        
        # Add process noise
        new_weight_var += self.Q[0, 0] * dt_days
        new_velocity_var += self.Q[1, 1] * dt_days
        
        # Update state
        self.state.weight = new_weight
        self.state.velocity = new_velocity
        self.state.weight_var = new_weight_var
        self.state.velocity_var = new_velocity_var
        self.state.weight_velocity_cov = new_cov
    
    def update(self, measurement: float) -> None:
        """
        Update state with new measurement
        
        Args:
            measurement: New weight measurement
        """
        # Innovation
        y = measurement - self.state.weight
        S = self.state.weight_var + self.measurement_noise
        
        # Kalman gain
        K_weight = self.state.weight_var / S
        K_velocity = self.state.weight_velocity_cov / S
        
        # Update state
        self.state.weight += K_weight * y
        self.state.velocity += K_velocity * y
        
        # Update covariance
        K = np.array([[K_weight], [K_velocity]])
        I_KH = np.eye(2) - K @ self.H
        
        new_cov = I_KH @ np.array([
            [self.state.weight_var, self.state.weight_velocity_cov],
            [self.state.weight_velocity_cov, self.state.velocity_var]
        ])
        
        self.state.weight_var = new_cov[0, 0]
        self.state.velocity_var = new_cov[1, 1]
        self.state.weight_velocity_cov = new_cov[0, 1]
    
    def forecast(self, days_ahead: float) -> Tuple[float, float]:
        """
        Forecast weight and uncertainty for a future date
        
        Args:
            days_ahead: Days in the future to forecast
            
        Returns:
            (forecasted_weight, forecasted_std)
        """
        # Work on a copy so we do not mutate the original state object
        original = self.state
        sim = KalmanState(
            weight=original.weight,
            velocity=original.velocity,
            weight_var=original.weight_var,
            velocity_var=original.velocity_var,
            weight_velocity_cov=original.weight_velocity_cov,
        )
        # Temporarily point to the copy, run predict, then restore
        self.state = sim
        self.predict(days_ahead)
        forecast_weight = float(self.state.weight)
        forecast_std = float(np.sqrt(max(self.state.weight_var, 0.0)))
        self.state = original
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
    
    prev_date = None
    
    for entry in entries:
        if prev_date is not None:
            # Predict forward to current date
            dt_days = (entry.entry_date - prev_date).days
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
        dates.append(entry.entry_date)
        
        prev_date = entry.entry_date
    
    return states, dates


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
    
    # Convert dates to days from first date for numerical interpolation
    t0 = dates[0]
    t_original = [(d - t0).days for d in dates]
    t_target = [(d - t0).days for d in target_dates]
    
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
    
    # Create cubic splines
    try:
        weight_spline = CubicSpline(t_original, weights, bc_type='natural')
        velocity_spline = CubicSpline(t_original, velocities, bc_type='natural')
        std_spline = CubicSpline(t_original, weight_stds, bc_type='natural')
        
        # Interpolate
        interpolated_weights = weight_spline(t_target)
        interpolated_velocities = velocity_spline(t_target)
        interpolated_stds = std_spline(t_target)
        
        # Ensure we have numpy arrays for consistent handling
        if not isinstance(interpolated_weights, np.ndarray):
            interpolated_weights = np.array(interpolated_weights)
        if not isinstance(interpolated_velocities, np.ndarray):
            interpolated_velocities = np.array(interpolated_velocities)
        if not isinstance(interpolated_stds, np.ndarray):
            interpolated_stds = np.array(interpolated_stds)
        
    except Exception as e:
        print(f"Warning: Cubic spline failed ({e}), falling back to linear interpolation")
        # Fallback to linear interpolation
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

    # Use datetimes for smooth plotting and collapse duplicate dates by averaging
    entry_datetimes = [datetime.combine(d, datetime.min.time()) for d in dates]

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


def create_kalman_plot(entries, 
                      states, 
                      dates,
                      output_path: str) -> None:
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
    
    # Build smooth dense curves exactly like the EMA spline approach (date-based x-axis)
    dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(states, dates)
    entry_datetimes = [datetime.combine(e.entry_date, datetime.min.time()) for e in entries]
    
    # Create plot with high-quality settings
    plt.figure(figsize=(12, 8), dpi=100)
    
    # Set matplotlib parameters for smooth curves (non-intrusive)
    plt.rcParams['savefig.dpi'] = 150
    
    # Plot confidence bands (95% confidence interval)
    confidence_level = 1.96  # 95% confidence
    upper_band = [m + confidence_level * s for m, s in zip(dense_means, dense_stds)]
    lower_band = [m - confidence_level * s for m, s in zip(dense_means, dense_stds)]
    
    # Fill confidence bands with anti-aliasing for smooth curves
    plt.fill_between(dense_datetimes, lower_band, upper_band, 
                     alpha=0.3, color='gray', label='95% Confidence Interval',
                     antialiased=True, linewidth=0)
    
    # Plot filtered state mean with anti-aliasing for smooth curves
    plt.plot(dense_datetimes, dense_means, '-', color='#ff7f0e', 
             linewidth=2, label='Kalman Filter Estimate', 
             antialiased=True, solid_capstyle='round')
    
    # Plot raw data
    raw_weights = [e.weight for e in entries]
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
        initial_weight=states[0].weight,
        initial_velocity=states[0].velocity
    )
    kf.state = latest_state
    
    # Forecast 1 week and 1 month ahead
    week_forecast, week_std = kf.forecast(7.0)
    month_forecast, month_std = kf.forecast(30.0)
    
    # Create stats text (mean/std from spline; slope from latest Kalman state)
    stats_text = f"""Current Estimate ({latest_date})
Weight: {current_mean:.2f} ± {1.96 * current_std:.2f}
Rate: {(7*latest_state.velocity):+.3f} lbs/week

Forecasts:
1 week: {week_forecast:.2f} ± {1.96 * week_std:.2f}
1 month: {month_forecast:.2f} ± {1.96 * month_std:.2f}"""
    
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
    plt.close()


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

    # Use smoothed Kalman mean and std over a dense date grid
    dense_datetimes, dense_means, dense_stds = compute_kalman_mean_std_spline(states, dates)
    if not dense_datetimes:
        return

    # Baseline defaults: if baseline weight not provided, use first Kalman mean
    if baseline_weight_lb is None:
        baseline_weight_lb = float(dense_means[0])

    # Scenario fractions
    s_low = 0.0
    s_mid = 0.10
    s_high = 0.20

    # Midline body fat from smoothed weights
    bf_mid = _compute_bodyfat_pct_from_weight(
        dense_means, dense_datetimes, baseline_weight_lb, baseline_lean_lb, s_mid
    )

    # Confidence band for midline: evaluate transformation at W±z*std
    z = 1.96
    w_lo = [max(1e-6, m - z * s) for m, s in zip(dense_means, dense_stds)]
    w_hi = [m + z * s for m, s in zip(dense_means, dense_stds)]
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
    entry_datetimes = [datetime.combine(e.entry_date, datetime.min.time()) for e in entries]
    entry_weights = [float(e.weight) for e in entries]
    entry_bf_mid = _compute_bodyfat_pct_from_weight(
        entry_weights, entry_datetimes, baseline_weight_lb, baseline_lean_lb, s_mid
    )

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8), dpi=100)

    # Confidence band around midline
    plt.fill_between(
        dense_datetimes, bf_band_lower, bf_band_upper,
        color="#cccccc", alpha=0.4, label="Midline 95% CI"
    )

    # Scenario bounds
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

    # Forecasts: transform Kalman weight forecasts through the body fat model (mid scenario)
    from kalman import WeightKalmanFilter, KalmanState
    kf = WeightKalmanFilter(initial_weight=states[0].weight, initial_velocity=states[0].velocity)
    # Use a copy of latest state
    kf.state = KalmanState(
        weight=states[-1].weight,
        velocity=states[-1].velocity,
        weight_var=states[-1].weight_var,
        velocity_var=states[-1].velocity_var,
        weight_velocity_cov=states[-1].weight_velocity_cov,
    )

    # 1 week forecast
    w_week, w_week_std = kf.forecast(7.0)
    bf_week_mid = _compute_bodyfat_pct_from_weight([w_week], [dense_datetimes[-1] + timedelta(days=7)],
                                                  baseline_weight_lb, baseline_lean_lb, s_mid)[0]
    bf_week_lo = _compute_bodyfat_pct_from_weight([max(1e-6, w_week - 1.96 * w_week_std)],
                                                  [dense_datetimes[-1] + timedelta(days=7)],
                                                  baseline_weight_lb, baseline_lean_lb, s_mid)[0]
    bf_week_hi = _compute_bodyfat_pct_from_weight([w_week + 1.96 * w_week_std],
                                                  [dense_datetimes[-1] + timedelta(days=7)],
                                                  baseline_weight_lb, baseline_lean_lb, s_mid)[0]
    bf_week_halfwidth = float(max(bf_week_hi - bf_week_mid, bf_week_mid - bf_week_lo))

    # 1 month forecast (30 days)
    w_month, w_month_std = kf.forecast(30.0)
    bf_month_mid = _compute_bodyfat_pct_from_weight([w_month], [dense_datetimes[-1] + timedelta(days=30)],
                                                   baseline_weight_lb, baseline_lean_lb, s_mid)[0]
    bf_month_lo = _compute_bodyfat_pct_from_weight([max(1e-6, w_month - 1.96 * w_month_std)],
                                                   [dense_datetimes[-1] + timedelta(days=30)],
                                                   baseline_weight_lb, baseline_lean_lb, s_mid)[0]
    bf_month_hi = _compute_bodyfat_pct_from_weight([w_month + 1.96 * w_month_std],
                                                   [dense_datetimes[-1] + timedelta(days=30)],
                                                   baseline_weight_lb, baseline_lean_lb, s_mid)[0]
    bf_month_halfwidth = float(max(bf_month_hi - bf_month_mid, bf_month_mid - bf_month_lo))

    # Stats box text
    stats_text = (
        f"Current Estimate ({latest_date})\n"
        f"Body Fat: {current_bf:.2f}% ± {current_halfwidth:.2f}%\n"
        f"Rate: {bf_slope_per_week:+.3f}%/week\n\n"
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
    plt.close()
