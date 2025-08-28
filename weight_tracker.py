#!/usr/bin/env python3
"""
Weight Tracker - Simple weight tracking with Kalman filtering
"""

import argparse
import csv
from datetime import date, datetime
from typing import List, Optional, Tuple
import numpy as np

# ---------------------------
# Data structures
# ---------------------------

class WeightEntry:
    def __init__(self, entry_date: date, weight: float):
        self.entry_date = entry_date
        self.weight = weight

    def __repr__(self) -> str:
        return f"WeightEntry({self.entry_date}, {self.weight})"

# ---------------------------
# Data loading and saving
# ---------------------------

def parse_date(date_str: str) -> date:
    """Parse date string in YYYY-MM-DD format"""
    return datetime.strptime(date_str, "%Y-%m-%d").date()

def load_entries(csv_path: str) -> List[WeightEntry]:
    """Load weight entries from CSV file"""
    entries = []
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    entry_date = parse_date(row['date'])
                    weight = float(row['weight'])
                    entries.append(WeightEntry(entry_date, weight))
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping invalid row: {row} ({e})")
    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
        return []
    
    # Sort by date
    entries.sort(key=lambda e: e.entry_date)
    return entries

def append_entry(csv_path: str, entry: WeightEntry) -> None:
    """Append a new weight entry to the CSV file"""
    # Check if file exists and has headers
    file_exists = False
    try:
        with open(csv_path, 'r') as f:
            first_line = f.readline().strip()
            file_exists = bool(first_line and first_line.startswith('date'))
    except FileNotFoundError:
        pass
    
    # Write entry
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['date', 'weight'])
        writer.writerow([entry.entry_date.strftime('%Y-%m-%d'), f"{entry.weight:.3f}"])

# ---------------------------
# Simple plotting
# ---------------------------

def render_simple_plot(entries: List[WeightEntry], output_path: str, no_display: bool = True, start_date: Optional[date] = None, end_date: Optional[date] = None) -> None:
    """Render a simple plot showing just the weight data points"""
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
    
    # Use datetimes for plotting
    dates = [datetime.combine(e.entry_date, datetime.min.time()) for e in filtered_entries]
    weights = [e.weight for e in filtered_entries]

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
        ys = [w for d, w in zip(dates, weights) if d.weekday() == wd]
        if xs:
            plt.scatter(xs, ys, s=36, color=weekday_colors[wd], label=weekday_names[wd], alpha=0.9, edgecolors="white", linewidths=0.5, zorder=3)

    plt.title("Weight Data")
    plt.xlabel("Date")
    plt.ylabel("Weight (lb)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    if no_display:
        plt.close()
    else:
        plt.show()

# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track weights with simple plotting and Kalman filtering.")
    parser.add_argument("--csv", default="weights.csv", help="Path to CSV file with columns: date, weight. Default: weights.csv")
    parser.add_argument("--add", action="append", default=[], help="Add an entry in the form YYYY-MM-DD:WEIGHT (can be used multiple times)")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate plot image")
    parser.add_argument("--output", default="weight_trend.png", help="Output plot image path. Default: weight_trend.png")
    parser.add_argument("--no-kalman-plot", action="store_true", help="Do not generate Kalman filter plot")
    parser.add_argument("--kalman-mode", choices=["filter", "smoother"], default="smoother",
                        help="Use forward Kalman filter ('filter') or RTS smoother ('smoother') for historical trend")
    # Body fat baseline parameters (used only when --kalman-plot is enabled)
    parser.add_argument("--bf-baseline-lean", type=float, default=150.0,
                        help="Baseline lean mass in lb (default: 150.0)")
    parser.add_argument("--bf-baseline-weight", type=float, default=None,
                        help="Baseline total weight in lb (default: first Kalman mean)")
    parser.add_argument("--no-display", action="store_true", help="Do not display plots in a GUI")
    parser.add_argument("--lbm-csv", type=str, default="lbm.csv",
                        help="Optional CSV with 'date,lbm' to drive body fat plot via interpolated LBM. Dates like YYYY-MM-DD.")
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

    # Print summary
    latest = entries[-1]
    print("=== Weight Data ===")
    print(f"Entries: {len(entries)} | Date range: {entries[0].entry_date} to {latest.entry_date}")
    print(f"Latest weight: {latest.weight:.2f} on {latest.entry_date}")

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

    if not args.no_plot:
        try:
            render_simple_plot(entries, args.output, no_display=args.no_display, start_date=start_date, end_date=end_date)
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
                from kalman import run_kalman_filter
                kalman_states, kalman_dates = run_kalman_filter(entries)
                plot_label = "Kalman Filter Estimate"
            
            if kalman_states:
                # Create Kalman filter plot
                from kalman import create_kalman_plot
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
                        lbm_csv=args.lbm_csv,
                    )
                    print("Body fat plot saved to: bodyfat_trend.png")
                except Exception as e:
                    print(f"Failed to generate body fat plot: {e}")
        except Exception as e:
            print(f"Failed to generate Kalman filter plot: {e}")


if __name__ == "__main__":
    main()


