#!/usr/bin/env python3

import argparse
import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Tuple, Optional, Dict

import numpy as np

# Import Kalman filter functionality
from kalman import run_kalman_filter, create_kalman_plot


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class WeightEntry:
    entry_date: date
    weight: float


# ---------------------------
# Utilities
# ---------------------------

DATE_FORMATS = [
    "%Y-%m-%d",  # 2025-08-19
    "%m/%d/%Y",  # 08/19/2025
    "%m/%d/%y",   # 8/19/25
    "%Y/%m/%d",  # 2025/08/19
    "%d-%b-%Y",  # 19-Aug-2025
]


def parse_date(value: str) -> date:
    last_err: Optional[Exception] = None
    v = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(v, fmt).date()
        except Exception as e:
            last_err = e
            continue
    # Try ISO 8601 fallback (handles YYYY-MM-DDTHH:MM:SS)
    try:
        return datetime.fromisoformat(v).date()
    except Exception:
        raise ValueError(f"Could not parse date: '{value}'. Tried formats: {', '.join(DATE_FORMATS)}") from last_err


def load_entries(csv_path: str) -> List[WeightEntry]:
    entries: List[WeightEntry] = []
    if not os.path.exists(csv_path):
        return entries
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Allow optional header; skip if first cell cannot be parsed as date
            try:
                d = parse_date(row[0])
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
    entries.sort(key=lambda e: e.entry_date)
    return entries


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
        writer.writerow([entry.entry_date.isoformat(), f"{entry.weight:.3f}"])


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
    prev_date: Optional[date] = None
    for e in entries:
        if ema is None:
            ema = e.weight
        else:
            delta_days = (e.entry_date - prev_date).days if prev_date else 1
            if delta_days <= 0:
                delta_days = 1
            eff_alpha = 1.0 - (1.0 - alpha_per_day) ** float(delta_days)
            ema = eff_alpha * e.weight + (1.0 - eff_alpha) * ema
        ema_values.append(float(ema))
        prev_date = e.entry_date
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

    dates = [e.entry_date for e in entries]
    weights_vals = [e.weight for e in entries]

    t0 = dates[0]
    t = np.array([(d - t0).days for d in dates], dtype=float)
    y = np.array(weights_vals, dtype=float)

    most_recent = dates[-1]
    age_days = np.array([(most_recent - d).days for d in dates], dtype=float)
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

def render_plot(entries: List[WeightEntry], ema_curve_dates: List[datetime], ema_curve_values: List[float], slope_per_day: float, intercept: float, output_path: str, no_display: bool = True, start_date: Optional[date] = None, end_date: Optional[date] = None) -> None:
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
            filtered_entries = [e for e in filtered_entries if e.entry_date >= start_date]
        if end_date:
            filtered_entries = [e for e in filtered_entries if e.entry_date <= end_date]
        
        if not filtered_entries:
            print(f"Warning: No data in specified date range {start_date} to {end_date}")
            return
    
    # Use datetimes for smooth plotting
    from datetime import timedelta
    dates = [datetime.combine(e.entry_date, datetime.min.time()) for e in filtered_entries]
    y = [e.weight for e in filtered_entries]

    # Create x values for regression line spanning the visible date range
    t0_dt = dates[0]
    t_vals = np.array([(d - t0_dt).total_seconds() / 86400.0 for d in dates], dtype=float)

    # Dense line for regression
    min_t, max_t = float(np.min(t_vals)), float(np.max(t_vals))
    t_dense = np.linspace(min_t, max_t, 200)
    y_reg = intercept + slope_per_day * t_dense
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
    
    # EMA curve and regression line
    plt.plot(ema_filtered_dates, ema_filtered_values, "-", label="7-day EMA (spline)", color="#ff7f0e", linewidth=2, zorder=2)
    plt.plot(dense_dates, y_reg, "--", label="Weighted regression", color="#000000", linewidth=2, zorder=1)

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

    # Prepare time axis in days from first timestamp, using datetimes
    entry_datetimes = [datetime.combine(e.entry_date, datetime.min.time()) for e in entries]

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
    parser.add_argument("--csv", default="weights.csv", help="Path to CSV file with columns: date, weight. Default: weights.csv")
    parser.add_argument("--add", action="append", default=[], help="Add an entry in the form YYYY-MM-DD:WEIGHT (can be used multiple times)")
    parser.add_argument("--ema-days", type=float, default=7.0, help="EMA span in days. Default: 7")
    parser.add_argument("--half-life-days", type=float, default=7.0, help="Half-life for time weighting in regression. Default: 7")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate plot image")
    parser.add_argument("--output", default="weight_trend.png", help="Output plot image path. Default: weight_trend.png")
    parser.add_argument("--no-kalman-plot", action="store_true", help="Do not generate Kalman filter plot")
    parser.add_argument("--kalman-mode", choices=["filter", "smoother"], default="smoother",
                        help="Use forward Kalman filter ('filter') or RTS smoother ('smoother') for historical trend")
    parser.add_argument("--print-table", action="store_true", help="Print table of date, weight, EMA")
    # Body fat baseline parameters (used only when --kalman-plot is enabled)
    parser.add_argument("--bf-baseline-lean", type=float, default=150.0,
                        help="Baseline lean mass in lb (default: 150.0)")
    parser.add_argument("--bf-baseline-weight", type=float, default=None,
                        help="Baseline total weight in lb (default: first Kalman mean)")
    parser.add_argument("--no-display", action="store_true", help="Do not display plots in a GUI")
    parser.add_argument("--start", type=str, help="Start date for plotting (YYYY-MM-DD format). If not specified, shows all data from beginning.")
    parser.add_argument("--end", type=str, help="End date for plotting (YYYY-MM-DD format). If not specified, shows all data to end.")
    return parser.parse_args()


def parse_add_arg(s: str) -> WeightEntry:
    try:
        date_part, weight_part = s.split(":", 1)
    except ValueError:
        raise ValueError("--add must be in the form YYYY-MM-DD:WEIGHT")
    d = parse_date(date_part.strip())
    try:
        w = float(weight_part.strip())
    except Exception:
        raise ValueError("Weight must be a number")
    return WeightEntry(d, w)


def main() -> None:
    args = parse_args()

    csv_path = args.csv

    # Handle additions first (persist them), then load full dataset
    for add_str in args.add:
        entry = parse_add_arg(add_str)
        append_entry(csv_path, entry)

    entries = load_entries(csv_path)

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
    print(f"Entries: {len(entries)} | Date range: {entries[0].entry_date} to {latest.entry_date}")
    print(f"Latest weight: {latest.weight:.2f} on {latest.entry_date}")
    print(f"7-day EMA: {latest_ema:.2f}")
    print(f"Estimated rate (weighted regression, half-life={args.half_life_days}d):")
    print(f"  per day:  {slope_per_day:+.4f}")
    print(f"  per week: {slope_per_week:+.3f}")
    print(f"  per month:{slope_per_month:+.3f}")
    print(f"Calorie deficit: {slope_per_day*3500:+.3f} calories/day")

    if args.print_table:
        print("\nDate,Weight,EMA")
        for e, ema in zip(entries, ema_values):
            print(f"{e.entry_date},{e.weight:.3f},{ema:.3f}")

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
            render_plot(entries, ema_dense_dates, ema_dense_values, slope_per_day, intercept, args.output, no_display=args.no_display, start_date=start_date, end_date=end_date)
            print(f"Plot saved to: {args.output}")
        except Exception as e:
            print(f"Failed to render plot: {e}")
    
    # Generate Kalman filter plot if requested
    if not args.no_kalman_plot:
        try:
            # Run Kalman algorithm per mode
            if args.kalman_mode == "smoother":
                from kalman import run_kalman_smoother
                kalman_states, kalman_dates = run_kalman_smoother(entries)
                plot_label = "Kalman RTS Smoother"
            else:
                kalman_states, kalman_dates = run_kalman_filter(entries)
                plot_label = "Kalman Filter Estimate"
            
            if kalman_states:
                # Create Kalman filter plot
                create_kalman_plot(entries, kalman_states, kalman_dates, args.output, no_display=args.no_display, label=plot_label, start_date=start_date, end_date=end_date)
                print("Kalman plot saved to: weight_trend.png")
                
                # Create body fat plot using Kalman smoothing
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
                    )
                    print("Body fat plot saved to: bodyfat_trend.png")
                except Exception as e:
                    print(f"Failed to generate body fat plot: {e}")
                
                # Print Kalman filter summary
                latest_kalman = kalman_states[-1]
                print(f"\n=== Kalman Filter Summary ===")
                print(f"Current weight estimate: {latest_kalman.weight:.2f} ± {1.96 * (latest_kalman.weight_var**0.5):.2f}")
                print(f"Current rate: {7*latest_kalman.velocity:+.3f} lbs/week")
                print(f"Calorie deficit: {latest_kalman.velocity*3500:+.3f} calories/day")
                
                # Calculate forecasts
                from kalman import WeightKalmanFilter
                kf = WeightKalmanFilter(initial_weight=kalman_states[0].weight)
                kf.state = latest_kalman
                
                week_forecast, week_std = kf.forecast(7.0)
                month_forecast, month_std = kf.forecast(30.0)
                
                print(f"1-week forecast: {week_forecast:.2f} ± {1.96 * week_std:.2f}")
                print(f"1-month forecast: {month_forecast:.2f} ± {1.96 * month_std:.2f}")
            else:
                print("No data available for Kalman filter")
        except Exception as e:
            print(f"Failed to generate Kalman filter plot: {e}")


if __name__ == "__main__":
    main()


