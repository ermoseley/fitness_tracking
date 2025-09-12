#!/usr/bin/env python3

import argparse
import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime, date, time
from typing import List, Tuple, Optional, Dict
from datetime import timedelta

import numpy as np

# Import Kalman filter functionality
from kalman import run_kalman_filter, create_kalman_plot


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class WeightEntry:
    entry_datetime: datetime
    weight: float


# ---------------------------
# Utilities
# ---------------------------

DEFAULT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(DEFAULT_BASE_DIR, "data")
DEFAULT_WEIGHTS_CSV = os.path.join(DEFAULT_DATA_DIR, "weights.csv")
DEFAULT_LBM_CSV = os.path.join(DEFAULT_DATA_DIR, "lbm.csv")

DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",     # 2025-08-19 14:30:00
    "%Y-%m-%d %H:%M",        # 2025-08-19 14:30
    "%Y-%m-%dT%H:%M:%S",     # 2025-08-19T14:30:00
    "%Y-%m-%dT%H:%M",        # 2025-08-19T14:30
    "%Y-%m-%dT%H:%M:%S.%f",  # 2025-08-19T14:30:00.123
    "%m/%d/%Y %H:%M:%S",     # 08/19/2025 14:30:00
    "%m/%d/%Y %H:%M",        # 08/19/2025 14:30
    "%m/%d/%y %H:%M:%S",     # 8/19/25 14:30:00
    "%m/%d/%y %H:%M",        # 8/19/25 14:30
    "%Y/%m/%d %H:%M:%S",     # 2025/08/19 14:30:00
    "%Y/%m/%d %H:%M",        # 2025/08/19 14:30
    "%d-%b-%Y %H:%M:%S",     # 19-Aug-2025 14:30:00
    "%d-%b-%Y %H:%M",        # 19-Aug-2025 14:30
    # Note: Date-only formats are handled by fallback logic with 9:00 AM default
]


def parse_datetime(value: str) -> datetime:
    """Parse a datetime string, supporting various formats including time of day"""
    last_err: Optional[Exception] = None
    v = value.strip()
    
    # Try all datetime formats
    for fmt in DATETIME_FORMATS:
        try:
            parsed = datetime.strptime(v, fmt)
            return parsed
        except Exception as e:
            last_err = e
            continue
    
    # Try ISO 8601 fallback (handles various ISO formats)
    try:
        parsed = datetime.fromisoformat(v)
        # If it's a date-only format (time is midnight), default to 9:00 AM
        if parsed.time() == datetime.min.time():
            return datetime.combine(parsed.date(), time(9, 0, 0))
        return parsed
    except Exception:
        pass
    
    # Try parsing as date-only formats and default to 9:00 AM
    date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d", "%d-%b-%Y"]
    for fmt in date_formats:
        try:
            date_only = datetime.strptime(v, fmt).date()
            return datetime.combine(date_only, time(9, 0, 0))
        except Exception:
            continue
    
    raise ValueError(f"Could not parse datetime: '{value}'. Supported formats include ISO 8601, YYYY-MM-DD HH:MM:SS, MM/DD/YYYY HH:MM:SS, etc.") from last_err


def parse_date(value: str) -> date:
    """Parse a date string (for backward compatibility)"""
    return parse_datetime(value).date()


def load_entries(csv_path: str) -> List[WeightEntry]:
    entries: List[WeightEntry] = []
    if not os.path.exists(csv_path):
        return entries
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Allow optional header; skip if first cell cannot be parsed as datetime
            try:
                d = parse_datetime(row[0])
            except Exception:
                # Probably a header row; skip
                continue
            try:
                raw_weight = str(row[1]).strip()
                # Extract the first numeric value (handles strings like '190.4%' or '190 lbs')
                match = re.search(r"[-+]?(?:\d+\.\d+|\d+)", raw_weight)
                if not match:
                    continue
                w = float(match.group(0))
            except Exception:
                continue
            entries.append(WeightEntry(d, w))
    entries.sort(key=lambda e: e.entry_datetime)
    return entries


def aggregate_entries(entries: List[WeightEntry], window_hours: float) -> List[WeightEntry]:
    """Aggregate entries within a rolling window into a single mean measurement.

    - Groups consecutive entries such that the span (max_time - min_time) within a group
      is ≤ window_hours.
    - The aggregated weight is the arithmetic mean of weights in the group.
    - The aggregated timestamp is the mean time of the group (by averaging offsets from
      the first timestamp to avoid timezone-related issues with naive datetimes).
    - If window_hours <= 0, returns entries unchanged (sorted by time).
    """
    if window_hours is None or window_hours <= 0:
        return sorted(entries, key=lambda e: e.entry_datetime)

    if not entries:
        return []

    sorted_entries = sorted(entries, key=lambda e: e.entry_datetime)
    window_seconds = float(window_hours) * 3600.0

    aggregated: List[WeightEntry] = []
    group: List[WeightEntry] = []

    for entry in sorted_entries:
        if not group:
            group = [entry]
            continue
        span_seconds = (entry.entry_datetime - group[0].entry_datetime).total_seconds()
        if span_seconds <= window_seconds:
            group.append(entry)
        else:
            # finalize current group
            if len(group) == 1:
                aggregated.append(group[0])
            else:
                # mean weight
                mean_w = float(np.mean([g.weight for g in group]))
                # mean time via offsets from first timestamp
                origin = group[0].entry_datetime
                offsets = [ (g.entry_datetime - origin).total_seconds() for g in group ]
                mean_offset = float(np.mean(offsets))
                mean_dt = origin + timedelta(seconds=mean_offset)
                aggregated.append(WeightEntry(mean_dt, mean_w))
            # start new group
            group = [entry]

    # finalize last group
    if group:
        if len(group) == 1:
            aggregated.append(group[0])
        else:
            mean_w = float(np.mean([g.weight for g in group]))
            origin = group[0].entry_datetime
            offsets = [ (g.entry_datetime - origin).total_seconds() for g in group ]
            mean_offset = float(np.mean(offsets))
            mean_dt = origin + timedelta(seconds=mean_offset)
            aggregated.append(WeightEntry(mean_dt, mean_w))

    return aggregated

def append_entry(csv_path: str, entry: WeightEntry) -> None:
    file_exists = os.path.exists(csv_path)
    # Create parent directories if needed
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    need_header = False
    need_leading_newline = False
    if not file_exists:
        need_header = True
    else:
        try:
            size = os.path.getsize(csv_path)
            if size == 0:
                need_header = True
            else:
                # Ensure last line ends with a newline before appending
                with open(csv_path, "rb") as frb:
                    try:
                        frb.seek(-1, os.SEEK_END)
                        last_byte = frb.read(1)
                        if last_byte not in (b"\n", b"\r"):
                            need_leading_newline = True
                    except OSError:
                        # File may be special; ignore
                        pass
        except Exception:
            pass

    with open(csv_path, "a", newline="") as f:
        if need_leading_newline:
            f.write("\n")
        writer = csv.writer(f)
        if need_header:
            writer.writerow(["date", "weight"])
        writer.writerow([entry.entry_datetime.isoformat(), f"{entry.weight:.3f}"])


# ---------------------------
# Analytics
# ---------------------------

def compute_time_aware_ema(entries: List[WeightEntry], span_days: float) -> List[float]:
    """
    Compute EMA with respect to irregular sampling by scaling alpha based on
    the number of days elapsed since the previous sample.

    alpha_per_day = 2/(span+1)
    for a gap of d days, effective alpha_d = 1 - (1 - alpha_per_day)**d
    """
    if not entries:
        return []
    alpha_per_day = 2.0 / (span_days + 1.0)
    ema_values: List[float] = []
    ema: Optional[float] = None
    prev_datetime: Optional[datetime] = None
    for e in entries:
        if ema is None:
            ema = e.weight
        else:
            delta_days = (e.entry_datetime - prev_datetime).total_seconds() / 86400.0 if prev_datetime else 1
            if delta_days <= 0:
                delta_days = 1
            eff_alpha = 1.0 - (1.0 - alpha_per_day) ** float(delta_days)
            ema = eff_alpha * e.weight + (1.0 - eff_alpha) * ema
        ema_values.append(float(ema))
        prev_datetime = e.entry_datetime
    return ema_values


def fit_time_weighted_linear_regression(entries: List[WeightEntry], half_life_days: float) -> Tuple[float, float]:
    """
    Fit y = intercept + slope * t using weighted least squares,
    where t is in days relative to the first date (shift does not affect slope),
    and weights decay exponentially with age relative to the most recent date.

    weight_i = np.exp(-age_days / half_life_days)
    Returns (slope_per_day, intercept_at_t0)
    """
    if len(entries) < 2:
        raise ValueError("Need at least 2 data points for regression")

    datetimes = [e.entry_datetime for e in entries]
    weights_vals = [e.weight for e in entries]

    t0 = datetimes[0]
    t = np.array([(d - t0).total_seconds() / 86400.0 for d in datetimes], dtype=float)
    y = np.array(weights_vals, dtype=float)

    most_recent = datetimes[-1]
    age_days = np.array([(most_recent - d).total_seconds() / 86400.0 for d in datetimes], dtype=float)
    if half_life_days <= 0:
        # No decay -> equal weights
        w = np.ones_like(age_days)
    else:
        w = np.exp(-age_days / float(half_life_days))

    # Build weighted normal equations
    W = np.diag(w)
    X = np.vstack([np.ones_like(t), t]).T  # columns: intercept, slope*t

    XtW = X.T @ W
    XtWX = XtW @ X
    XtWy = XtW @ y

    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse in case of singularity
        beta = np.linalg.pinv(XtWX) @ XtWy

    intercept, slope = float(beta[0]), float(beta[1])
    return slope, intercept


# ---------------------------
# Plotting
# ---------------------------

def render_plot(entries: List[WeightEntry], ema_curve_dates: List[datetime], ema_curve_values: List[float], slope_per_day: float, intercept: float, output_path: str, no_display: bool = True, start_date: Optional[date] = None, end_date: Optional[date] = None, ci_multiplier: float = 1.96, enable_forecast: bool = True) -> None:
    import matplotlib
    # Use a non-interactive backend only if we are not displaying
    if no_display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not entries:
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
    
    # Use datetimes directly for plotting
    dates = [e.entry_datetime for e in filtered_entries]
    y = [e.weight for e in filtered_entries]

    # Create x values for regression line spanning the visible date range
    t0_dt = dates[0]
    t_vals = np.array([(d - t0_dt).total_seconds() / 86400.0 for d in dates], dtype=float)

    # Dense line for regression
    min_t, max_t = float(np.min(t_vals)), float(np.max(t_vals))
    t_dense = np.linspace(min_t, max_t, 200)
    y_reg = intercept + slope_per_day * t_dense
    from datetime import timedelta
    dense_dates = [t0_dt + timedelta(days=float(td)) for td in t_dense]

    plt.figure(figsize=(10, 6))

    # Scatter weights colored by weekday
    weekday_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    weekday_colors = {
        0: "tab:blue",
        1: "tab:green",
        2: "tab:red",
        3: "tab:purple",
        4: "tab:brown",
        5: "tab:pink",
        6: "tab:gray",
    }
    for wd in range(7):
        xs = [d for d in dates if d.weekday() == wd]
        ys = [yv for d, yv in zip(dates, y) if d.weekday() == wd]
        if xs:
            plt.scatter(xs, ys, s=36, color=weekday_colors[wd], label=weekday_names[wd], alpha=0.9, edgecolors="white", linewidths=0.5, zorder=3)

    # Filter EMA curve data to match the date range
    if start_date or end_date:
        ema_filtered_dates = []
        ema_filtered_values = []
        for dt, val in zip(ema_curve_dates, ema_curve_values):
            dt_date = dt.date()
            if start_date and dt_date < start_date:
                continue
            if end_date and dt_date > end_date:
                continue
            ema_filtered_dates.append(dt)
            ema_filtered_values.append(val)
    else:
        ema_filtered_dates = ema_curve_dates
        ema_filtered_values = ema_curve_values
    
    # EMA curve and regression line (filter regression to visible date range if needed)
    plt.plot(ema_filtered_dates, ema_filtered_values, "-", label="7-day EMA (spline)", color="#ff7f0e", linewidth=2, zorder=2)
    if start_date or end_date:
        dense_filtered_dates = []
        dense_filtered_y = []
        for d, y in zip(dense_dates, y_reg):
            d_date = d.date()
            if start_date and d_date < start_date:
                continue
            if end_date and d_date > end_date:
                continue
            dense_filtered_dates.append(d)
            dense_filtered_y.append(y)
        plt.plot(dense_filtered_dates, dense_filtered_y, "--", label="Weighted regression", color="#000000", linewidth=2, zorder=1)
    else:
        plt.plot(dense_dates, y_reg, "--", label="Weighted regression", color="#000000", linewidth=2, zorder=1)

    # Add forecasting fan if enabled
    if enable_forecast and not (start_date or end_date):  # Only show forecast when not filtering by date
        # Calculate forecast parameters
        last_date = max(dates)
        last_weight = y[-1]
        
        # Calculate historical variance for uncertainty estimation
        if len(y) > 1:
            # Use the variance of the last 30 days or all data if less than 30 days
            recent_entries = y[-min(30, len(y)):]
            historical_std = np.std(recent_entries)
        else:
            historical_std = 0.5  # Default uncertainty
        
        # Create forecast dates (1 month = 30 days)
        forecast_days = 30
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        
        # Calculate forecast values using linear trend
        forecast_weights = [last_weight + slope_per_day * i for i in range(1, forecast_days + 1)]
        
        # Calculate uncertainty bands (growing uncertainty over time)
        forecast_stds = [historical_std * np.sqrt(1 + i/30) for i in range(1, forecast_days + 1)]
        
        # Create confidence intervals
        forecast_upper = [w + ci_multiplier * std for w, std in zip(forecast_weights, forecast_stds)]
        forecast_lower = [w - ci_multiplier * std for w, std in zip(forecast_weights, forecast_stds)]
        
        # Plot forecast fan
        plt.fill_between(forecast_dates, forecast_lower, forecast_upper, 
                        alpha=0.2, color='gray', label=f'Forecast ±{ci_multiplier:.1f}σ (1 month)')
        
        # Plot forecast mean line
        plt.plot(forecast_dates, forecast_weights, '--', color='gray', linewidth=1.5, 
                alpha=0.8, label='Forecast trend')

    plt.title("Weight Trend")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.grid(True, alpha=0.3)
    plt.legend()
    # Add estimated rate annotation (per day/week/month)
    slope_week = slope_per_day * 7.0
    slope_month = slope_per_day * 30.0
    ax = plt.gca()
    rate_text = f"Rate\n{slope_per_day:+.4f}/day\n{slope_week:+.3f}/week\n{slope_month:+.3f}/month"
    ax.text(0.02, 0.02, rate_text, transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"), zorder=10)

    # Set x-axis limits based on date filtering
    # Set x-axis using filtered dense range with padding so bands stay inside
    dense_min = min(dense_filtered_dates) if (start_date or end_date) and 'dense_filtered_dates' in locals() and dense_filtered_dates else (min(dense_dates) if dense_dates else (min(ema_filtered_dates) if ema_filtered_dates else None))
    dense_max = max(dense_filtered_dates) if (start_date or end_date) and 'dense_filtered_dates' in locals() and dense_filtered_dates else (max(dense_dates) if dense_dates else (max(ema_filtered_dates) if ema_filtered_dates else None))
    if start_date or end_date:
        left = datetime.combine(start_date, datetime.min.time()) if start_date else dense_min
        right = datetime.combine(end_date, datetime.max.time()) if end_date else dense_max
    else:
        left, right = dense_min, dense_max
    if left is not None and right is not None and left <= right:
        span = right - left
        from datetime import timedelta as _td
        pad = max(span * 0.02, _td(days=1))
        left_pad = left if start_date else (left - pad)
        right_pad = right if end_date else (right + pad)
        plt.xlim(left=left_pad, right=right_pad)

    # Autoscale Y after x-limits to fit visible data
    ax = plt.gca()
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    if no_display:
        plt.close()
    else:
        plt.show()


# ---------------------------
# EMA on entries + spline interpolation for smooth curve
# ---------------------------

def compute_ema_and_spline(entries: List[WeightEntry], span_days: float) -> Tuple[List[datetime], List[float], List[float]]:
    """
    Compute standard EMA at the entry dates only, then generate a smooth, dense
    curve by cubic interpolation between those EMA points. Returns:
    (dense_datetime, dense_ema_values, ema_at_entry_dates)
    """
    if not entries:
        return [], [], []

    # Compute standard time-aware EMA at entry points (irregular gaps handled)
    ema_at_entries = compute_time_aware_ema(entries, span_days=span_days)

    # Use entry datetimes directly
    entry_datetimes = [e.entry_datetime for e in entries]

    # Collapse duplicate dates by averaging EMA values for that date (ensures strictly increasing x)
    from collections import defaultdict
    dt_to_emas: Dict[datetime, List[float]] = defaultdict(list)
    for dt, ema in zip(entry_datetimes, ema_at_entries):
        dt_to_emas[dt].append(float(ema))
    unique_datetimes = sorted(dt_to_emas.keys())
    ema_vals_unique = [float(np.mean(dt_to_emas[dt])) for dt in unique_datetimes]

    t0 = unique_datetimes[0]
    t_entry_days = np.array([(dt - t0).total_seconds() / 86400.0 for dt in unique_datetimes], dtype=float)
    ema_vals = np.array(ema_vals_unique, dtype=float)

    # Build a dense time grid for smooth plotting
    if len(t_entry_days) == 1:
        return [unique_datetimes[0]], [float(ema_vals[0])], [float(ema_vals[0])]

    min_t = float(np.min(t_entry_days))
    max_t = float(np.max(t_entry_days))
    dense_t = np.linspace(min_t, max_t, 5000)

    # Cubic interpolation between EMA points
    try:
        print("Importing CubicSpline")
        from scipy.interpolate import CubicSpline  # type: ignore
        spline = CubicSpline(t_entry_days, ema_vals, bc_type="natural")
        ema_dense = spline(dense_t)
    except Exception:
        # Fallback: PCHIP if available, else linear
        try:
            print("Importing PchipInterpolator")
            from scipy.interpolate import PchipInterpolator  # type: ignore
            pchip = PchipInterpolator(t_entry_days, ema_vals)
            ema_dense = pchip(dense_t)
        except Exception:
            print("Importing np.interp")
            ema_dense = np.interp(dense_t, t_entry_days, ema_vals)

    from datetime import timedelta
    dense_datetimes = [t0 + timedelta(days=float(td)) for td in dense_t]

    return dense_datetimes, [float(v) for v in ema_dense], [float(v) for v in ema_at_entries]


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track weights, compute EMA and time-weighted linear regression trend.")
    parser.add_argument("--csv", default=DEFAULT_WEIGHTS_CSV, help="Path to CSV file with columns: date, weight. Default: data/weights.csv")
    parser.add_argument("--add", action="append", default=[], help="Add an entry in the form YYYY-MM-DD:WEIGHT (can be used multiple times)")
    parser.add_argument("--ema-days", type=float, default=7.0, help="EMA span in days. Default: 7")
    parser.add_argument("--half-life-days", type=float, default=7.0, help="Half-life for time weighting in regression. Default: 7")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate plot image")
    parser.add_argument("--output", default="weight_trend.png", help="Output plot image path. Default: weight_trend.png")
    parser.add_argument("--no-kalman-plot", action="store_true", help="Do not generate Kalman filter plot")
    parser.add_argument("--no-bodyfat-plot", action="store_true", help="Do not generate body fat plot")
    parser.add_argument("--no-bmi-plot", action="store_true", help="Do not generate BMI plot")
    parser.add_argument("--no-ffmi-plot", action="store_true", help="Do not generate FFMI plot")
    parser.add_argument("--kalman-mode", choices=["filter", "smoother"], default="smoother",
                        help="Use forward Kalman filter ('filter') or RTS smoother ('smoother') for historical trend")
    parser.add_argument("--print-table", action="store_true", help="Print table of date, weight, EMA")
    # Body fat baseline parameters (used only when --kalman-plot is enabled)
    parser.add_argument("--bf-baseline-lean", type=float, default=150.0,
                        help="Baseline lean mass in lb (default: 150.0)")
    parser.add_argument("--bf-baseline-weight", type=float, default=None,
                        help="Baseline total weight in lb (default: first Kalman mean)")
    parser.add_argument("--no-display", action="store_true", help="Do not display plots in a GUI")
    parser.add_argument("--lbm-csv", type=str, default=DEFAULT_LBM_CSV,
                        help="Optional CSV with 'date,lbm' to drive body fat plot via interpolated LBM. Default: data/lbm.csv")
    parser.add_argument("--start", type=str, help="Start date for plotting (YYYY-MM-DD format). If not specified, shows all data from beginning.")
    parser.add_argument("--end", type=str, help="End date for plotting (YYYY-MM-DD format). If not specified, shows all data to end.")
    parser.add_argument("--residuals-histogram", action="store_true", help="Generate residuals histogram and normality test")
    parser.add_argument("--confidence-interval", choices=["1σ", "95%"], default="95%", help="Confidence interval level for all plots and calculations")
    parser.add_argument("--aggregation-hours", type=float, default=3.0, help="Aggregation window in hours (0 to disable). Default: 3")
    return parser.parse_args()


def get_confidence_multiplier(ci_choice: str) -> float:
    """Convert confidence interval choice to multiplier for standard deviations"""
    if ci_choice == "1σ":
        return 1.0
    elif ci_choice == "95%":
        return 1.96
    else:
        raise ValueError(f"Unknown confidence interval choice: {ci_choice}")


def parse_add_arg(s: str) -> WeightEntry:
    # Split on the last colon so HH:MM:SS works
    pos = s.rfind(":")
    if pos == -1:
        raise ValueError("--add must be in the form YYYY-MM-DD[THH:MM:SS]:WEIGHT")
    datetime_part, weight_part = s[:pos], s[pos+1:]
    d = parse_datetime(datetime_part.strip())
    try:
        w = float(weight_part.strip())
    except Exception:
        raise ValueError("Weight must be a number")
    return WeightEntry(d, w)


def main() -> None:
    args = parse_args()

    csv_path = args.csv

    # Ensure data directory exists for default paths
    try:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    except Exception:
        pass

    # Handle additions first (persist them), then load full dataset
    if args.add:
        for add_str in args.add:
            entry = parse_add_arg(add_str)
            append_entry(csv_path, entry)
        # If only adding and no plots requested, exit cleanly
        if args.no_plot and args.no_kalman_plot:
            return

    entries = load_entries(csv_path)
    # Apply aggregation window prior to all downstream computations
    agg_hours = float(args.aggregation_hours) if hasattr(args, 'aggregation_hours') else 3.0
    if agg_hours > 0:
        entries = aggregate_entries(entries, agg_hours)

    if not entries:
        print("No data found. Use --add YYYY-MM-DD:WEIGHT to add entries or create the CSV.")
        return

    # Compute EMA at entries and spline-interpolate for smooth curve
    ema_dense_dates, ema_dense_values, ema_values = compute_ema_and_spline(entries, span_days=args.ema_days)

    slope_per_day, intercept = fit_time_weighted_linear_regression(entries, half_life_days=args.half_life_days)
    slope_per_week = slope_per_day * 7.0
    slope_per_month = slope_per_day * 30.0

    # Print summary
    latest = entries[-1]
    latest_ema = ema_values[-1]
    print("=== Weight Trend ===")
    print(f"Entries: {len(entries)} | Date range: {entries[0].entry_datetime} to {latest.entry_datetime}")
    print(f"Latest weight: {latest.weight:.2f} on {latest.entry_datetime}")
    print(f"7-day EMA: {latest_ema:.2f}")
    print(f"Estimated rate (weighted regression, half-life={args.half_life_days}d):")
    print(f"  per day:  {slope_per_day:+.4f}")
    print(f"  per week: {slope_per_week:+.3f}")
    print(f"  per month:{slope_per_month:+.3f}")
    print(f"Calorie deficit: {slope_per_day*3500:+.3f} calories/day")

    if args.print_table:
        print("\nDateTime,Weight,EMA")
        for e, ema in zip(entries, ema_values):
            print(f"{e.entry_datetime},{e.weight:.3f},{ema:.3f}")

    # Parse date range arguments
    start_date = None
    end_date = None
    if args.start:
        try:
            start_date = parse_date(args.start)
        except ValueError as e:
            print(f"Warning: Could not parse start date '{args.start}': {e}")
    if args.end:
        try:
            end_date = parse_date(args.end)
        except ValueError as e:
            print(f"Warning: Could not parse end date '{args.end}': {e}")

    if ((not args.no_plot) and args.no_kalman_plot):
        try:
            ci_multiplier = get_confidence_multiplier(args.confidence_interval)
            render_plot(entries, ema_dense_dates, ema_dense_values, slope_per_day, intercept, args.output, no_display=args.no_display, start_date=start_date, end_date=end_date, ci_multiplier=ci_multiplier, enable_forecast=True)
            print(f"Plot saved to: {args.output}")
        except Exception as e:
            print(f"Failed to render plot: {e}")
    
    # Generate plots that require Kalman filtering
    kalman_states = None
    kalman_dates = None
    ci_multiplier = get_confidence_multiplier(args.confidence_interval)
    
    # Run Kalman algorithm if any Kalman-dependent plots are requested
    # Note: We run Kalman if weight plot OR any other Kalman-dependent plots are enabled
    print(f"DEBUG: Plot flags - no_kalman_plot: {args.no_kalman_plot}, no_bodyfat_plot: {args.no_bodyfat_plot}, no_bmi_plot: {args.no_bmi_plot}, no_ffmi_plot: {args.no_ffmi_plot}")
    if not args.no_kalman_plot or not args.no_bodyfat_plot or not args.no_bmi_plot or not args.no_ffmi_plot:
        print("DEBUG: Running Kalman algorithm...")
        try:
            # Run Kalman algorithm per mode
            if args.kalman_mode == "smoother":
                from kalman import run_kalman_smoother
                kalman_states, kalman_dates = run_kalman_smoother(entries)
                plot_label = "Kalman RTS Smoother"
            else:
                kalman_states, kalman_dates = run_kalman_filter(entries)
                plot_label = "Kalman Filter Estimate"
            print(f"DEBUG: Kalman algorithm completed successfully. States: {len(kalman_states) if kalman_states else 0}")
        except Exception as e:
            print(f"Failed to run Kalman filter: {e}")
            kalman_states = None
            kalman_dates = None
    else:
        print("DEBUG: Kalman algorithm not run - no Kalman-dependent plots enabled")
    
    if kalman_states:
        # Create Kalman filter plot (if not disabled)
        if not args.no_kalman_plot:
            create_kalman_plot(entries, kalman_states, kalman_dates, args.output, no_display=args.no_display, label=plot_label, start_date=start_date, end_date=end_date, ci_multiplier=ci_multiplier, enable_forecast=True)
            print("Kalman plot saved to: weight_trend.png")
        
        # Create body fat plot using Kalman smoothing (if not disabled)
        if not args.no_bodyfat_plot:
            from kalman import create_bodyfat_plot_from_kalman
            try:
                create_bodyfat_plot_from_kalman(
                    entries,
                    kalman_states,
                    kalman_dates,
                    baseline_weight_lb=args.bf_baseline_weight,
                    baseline_lean_lb=args.bf_baseline_lean,
                    output_path="bodyfat_trend.png",
                    no_display=args.no_display,
                    start_date=start_date,
                    end_date=end_date,
                    lbm_csv=args.lbm_csv,
                    ci_multiplier=ci_multiplier,
                    enable_forecast=True,
                )
                print("Body fat plot saved to: bodyfat_trend.png")
            except Exception as e:
                print(f"Failed to generate body fat plot: {e}")
        
        # Create BMI plot using Kalman smoothing (if not disabled)
        if not args.no_bmi_plot:
            from kalman import create_bmi_plot_from_kalman
            try:
                create_bmi_plot_from_kalman(
                    entries,
                    kalman_states,
                    kalman_dates,
                    height_file="data/height.txt",
                    output_path="bmi_trend.png",
                    no_display=args.no_display,
                    start_date=start_date,
                    end_date=end_date,
                    ci_multiplier=ci_multiplier,
                    enable_forecast=True,
                )
                print("BMI plot saved to: bmi_trend.png")
            except Exception as e:
                print(f"Failed to generate BMI plot: {e}")
        
        # Create FFMI plot using Kalman smoothing (if not disabled)
        if not args.no_ffmi_plot:
            from kalman import create_ffmi_plot_from_kalman
            try:
                create_ffmi_plot_from_kalman(
                    entries,
                    kalman_states,
                    kalman_dates,
                    height_file="data/height.txt",
                    output_path="ffmi_trend.png",
                    no_display=args.no_display,
                    start_date=start_date,
                    end_date=end_date,
                    lbm_csv=args.lbm_csv,
                    ci_multiplier=ci_multiplier,
                    enable_forecast=True,
                )
                print("FFMI plot saved to: ffmi_trend.png")
            except Exception as e:
                print(f"Failed to generate FFMI plot: {e}")
        
        # Print Kalman filter summary
        latest_kalman = kalman_states[-1]
        print(f"\n=== Kalman Filter Summary ===")
        print(f"Current weight estimate: {latest_kalman.weight:.2f} ± {ci_multiplier * (latest_kalman.weight_var**0.5):.2f}")
        
        # Calculate velocity uncertainty
        velocity_std = (latest_kalman.velocity_var**0.5)
        velocity_ci = ci_multiplier * velocity_std
        velocity_per_week = 7 * latest_kalman.velocity
        velocity_ci_per_week = 7 * velocity_ci
        print(f"Current rate: {velocity_per_week:+.3f} ± {velocity_ci_per_week:.3f} lbs/week")
        print(f"Calorie deficit: {latest_kalman.velocity*3500:+.3f} calories/day")
        
        # Calculate forecasts
        from kalman import WeightKalmanFilter
        # Initialize KF to the latest state exactly as in the plot code
        kf = WeightKalmanFilter(
            initial_weight=latest_kalman.weight,
            initial_velocity=latest_kalman.velocity,
            initial_weight_var=latest_kalman.weight_var,
            initial_velocity_var=latest_kalman.velocity_var,
        )
        # Set full state and covariance including off-diagonal
        kf.x = np.array([latest_kalman.weight, latest_kalman.velocity], dtype=float)
        kf.P = np.array([
            [latest_kalman.weight_var, latest_kalman.weight_velocity_cov],
            [latest_kalman.weight_velocity_cov, latest_kalman.velocity_var],
        ], dtype=float)

        week_forecast, week_std = kf.forecast(7.0)
        month_forecast, month_std = kf.forecast(30.0)
        
        print(f"1-week forecast: {week_forecast:.2f} ± {ci_multiplier * week_std:.2f}")
        print(f"1-month forecast: {month_forecast:.2f} ± {ci_multiplier * month_std:.2f}")
        
        # Generate residuals histogram if requested
        if args.residuals_histogram:
            try:
                from kalman import compute_residuals, create_residuals_histogram
                residuals = compute_residuals(entries, kalman_states, kalman_dates, start_date, end_date)
                if residuals:
                    mean_res, std_res, p_value = create_residuals_histogram(
                        residuals, 
                        output_path="residuals_histogram.png",
                        no_display=args.no_display,
                        start_date=start_date,
                        end_date=end_date,
                        ci_multiplier=ci_multiplier
                    )
                    print(f"Residuals histogram saved to: residuals_histogram.png")
                else:
                    print("No residuals data available for histogram")
            except Exception as e:
                print(f"Failed to generate residuals histogram: {e}")
    else:
        print("No data available for Kalman filter")


if __name__ == "__main__":
    main()


