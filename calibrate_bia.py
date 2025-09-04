#!/usr/bin/env python3
"""
BIA to DEXA Calibration Tool

Calibrates daily bioelectrical impedance (BIA) body fat readings to DEXA scans
using linear mappings on fat mass (lb), not percent.

Usage:
    python calibrate_bia.py --bia data/bf_impedance_cleaned.csv --dxa data/lbm.csv --weights data/weights.csv --out data/calibrated_bf.csv
"""

import argparse
import csv
import json
import os
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

def parse_datetime(date_str: str) -> datetime:
    """Parse various datetime formats to datetime object."""
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S", 
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    
    # Try ISO format as fallback
    try:
        return datetime.fromisoformat(date_str.strip())
    except ValueError:
        raise ValueError(f"Could not parse date: {date_str}")

def load_bia_data(filepath: str) -> List[Dict]:
    """Load BIA data from CSV file."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                date = parse_datetime(row['date'])
                bf_pct = float(row['body_fat_pct'])
                athlete_mode = int(row['athlete_mode'])
                data.append({
                    'date': date,
                    'bf_pct': bf_pct,
                    'athlete_mode': athlete_mode
                })
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping invalid BIA row: {row} - {e}")
                continue
    return sorted(data, key=lambda x: x['date'])

def load_weights_data(filepath: str) -> List[Dict]:
    """Load weight data from CSV file."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                date = parse_datetime(row['date'])
                weight = float(row['weight'])
                data.append({
                    'date': date,
                    'weight': weight
                })
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping invalid weight row: {row} - {e}")
                continue
    return sorted(data, key=lambda x: x['date'])

def load_dexa_data(filepath: str) -> List[Dict]:
    """Load DEXA LBM data and convert to fat mass."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                date = parse_datetime(row['date'])
                lbm = float(row['lbm'])
                data.append({
                    'date': date,
                    'lbm': lbm
                })
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping invalid DEXA row: {row} - {e}")
                continue
    return sorted(data, key=lambda x: x['date'])

def interpolate_weight(date: datetime, weights_data: List[Dict]) -> Optional[float]:
    """Interpolate weight for a given date from weights data."""
    if not weights_data:
        return None
    
    # Find closest weight measurement
    min_diff = float('inf')
    closest_weight = None
    
    for w_data in weights_data:
        diff = abs((date - w_data['date']).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest_weight = w_data['weight']
    
    return closest_weight

def moving_median(data: List[float], window: int) -> List[float]:
    """Calculate moving median for smoothing."""
    if window <= 1:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        end = i + 1
        window_data = data[start:end]
        smoothed.append(np.median(window_data))
    
    return smoothed

def find_anchor_pairs(bia_data: List[Dict], dexa_data: List[Dict], weights_data: List[Dict], 
                     anchor_window_days: int = 3) -> List[Dict]:
    """Find BIA-DEXA anchor pairs within the specified time window."""
    anchor_pairs = []
    
    for dexa in dexa_data:
        dexa_date = dexa['date']
        dexa_lbm = dexa['lbm']
        
        # Get weight for DEXA date to calculate total weight
        dexa_weight = interpolate_weight(dexa_date, weights_data)
        if dexa_weight is None:
            print(f"Warning: No weight data found for DEXA date {dexa_date}")
            continue
        
        # Calculate DEXA fat mass
        dexa_fat_mass = dexa_weight - dexa_lbm
        
        # Find closest BIA measurement within window
        best_bia = None
        min_diff = float('inf')
        
        for bia in bia_data:
            diff_days = abs((bia['date'] - dexa_date).total_seconds() / 86400)
            if diff_days <= anchor_window_days:
                if diff_days < min_diff:
                    min_diff = diff_days
                    best_bia = bia
        
        if best_bia:
            # Get weight for BIA date
            bia_weight = interpolate_weight(best_bia['date'], weights_data)
            if bia_weight is None:
                print(f"Warning: No weight data found for BIA date {best_bia['date']}")
                continue
            
            # Calculate BIA fat mass
            bia_fat_mass = bia_weight * best_bia['bf_pct'] / 100.0
            
            anchor_pairs.append({
                'dexa_date': dexa_date,
                'dexa_fat_mass': dexa_fat_mass,
                'bia_date': best_bia['date'],
                'bia_fat_mass': bia_fat_mass,
                'bia_bf_pct': best_bia['bf_pct'],
                'bia_athlete_mode': best_bia['athlete_mode'],
                'weight': bia_weight,
                'time_diff_days': min_diff
            })
            print(f"Anchor pair: DEXA {dexa_date.date()} (FM={dexa_fat_mass:.2f}lb) <-> BIA {best_bia['date'].date()} (FM={bia_fat_mass:.2f}lb, mode={best_bia['athlete_mode']})")
        else:
            print(f"Warning: No BIA measurement found within {anchor_window_days} days of DEXA {dexa_date.date()}")
    
    return anchor_pairs

def calibrate_athlete_mode(anchor_pairs: List[Dict]) -> Tuple[float, float]:
    """Calibrate athlete mode BIA measurements to DEXA."""
    athlete_pairs = [p for p in anchor_pairs if p['bia_athlete_mode'] == 1]
    
    if len(athlete_pairs) < 1:
        print("Warning: No athlete mode anchor pairs found, using identity calibration")
        return 0.0, 1.0
    
    if len(athlete_pairs) == 1:
        # Baseline offset method
        alpha = athlete_pairs[0]['dexa_fat_mass'] - athlete_pairs[0]['bia_fat_mass']
        beta = 1.0
        print(f"Athlete calibration (baseline offset): α={alpha:.3f}, β={beta:.3f}")
    else:
        # Linear regression
        bia_fm = np.array([p['bia_fat_mass'] for p in athlete_pairs])
        dexa_fm = np.array([p['dexa_fat_mass'] for p in athlete_pairs])
        
        # OLS regression
        beta, alpha, r_value, p_value, std_err = stats.linregress(bia_fm, dexa_fm)
        print(f"Athlete calibration (OLS): α={alpha:.3f}, β={beta:.3f}, R²={r_value**2:.3f}")
    
    return alpha, beta

def calibrate_non_athlete_mode(anchor_pairs: List[Dict]) -> Tuple[float, float]:
    """Calibrate non-athlete mode BIA measurements to athlete mode equivalent."""
    non_athlete_pairs = [p for p in anchor_pairs if p['bia_athlete_mode'] == 0]
    
    if len(non_athlete_pairs) < 1:
        print("Warning: No non-athlete mode anchor pairs found, using identity calibration")
        return 0.0, 1.0
    
    # For non-athlete mode, we'll use a simple scaling factor
    # This is a simplified approach - in practice, you might want more sophisticated calibration
    bia_fm = np.array([p['bia_fat_mass'] for p in non_athlete_pairs])
    dexa_fm = np.array([p['dexa_fat_mass'] for p in non_athlete_pairs])
    
    if len(non_athlete_pairs) == 1:
        # Simple scaling
        beta = dexa_fm[0] / bia_fm[0] if bia_fm[0] != 0 else 1.0
        alpha = 0.0
    else:
        # Linear regression
        beta, alpha, r_value, p_value, std_err = stats.linregress(bia_fm, dexa_fm)
    
    print(f"Non-athlete calibration: α={alpha:.3f}, β={beta:.3f}")
    return alpha, beta

def apply_calibration(bia_data: List[Dict], weights_data: List[Dict], 
                     athlete_alpha: float, athlete_beta: float,
                     non_athlete_alpha: float, non_athlete_beta: float,
                     smooth_window: int = 7) -> List[Dict]:
    """Apply calibration to all BIA data."""
    calibrated_data = []
    
    # Separate athlete and non-athlete data for smoothing
    athlete_data = [d for d in bia_data if d['athlete_mode'] == 1]
    non_athlete_data = [d for d in bia_data if d['athlete_mode'] == 0]
    
    # Smooth fat mass for each mode separately
    if smooth_window > 1:
        athlete_fm = [interpolate_weight(d['date'], weights_data) * d['bf_pct'] / 100.0 
                     for d in athlete_data if interpolate_weight(d['date'], weights_data) is not None]
        non_athlete_fm = [interpolate_weight(d['date'], weights_data) * d['bf_pct'] / 100.0 
                         for d in non_athlete_data if interpolate_weight(d['date'], weights_data) is not None]
        
        athlete_smooth_fm = moving_median(athlete_fm, smooth_window) if athlete_fm else []
        non_athlete_smooth_fm = moving_median(non_athlete_fm, smooth_window) if non_athlete_fm else []
    else:
        athlete_smooth_fm = []
        non_athlete_smooth_fm = []
    
    # Process all data
    athlete_idx = 0
    non_athlete_idx = 0
    
    for bia in bia_data:
        weight = interpolate_weight(bia['date'], weights_data)
        if weight is None:
            continue
        
        bia_fm = weight * bia['bf_pct'] / 100.0
        
        # Apply appropriate calibration
        if bia['athlete_mode'] == 1:
            if smooth_window > 1 and athlete_idx < len(athlete_smooth_fm):
                bia_smooth_fm = athlete_smooth_fm[athlete_idx]
                athlete_idx += 1
            else:
                bia_smooth_fm = bia_fm
            
            cal_fm = athlete_alpha + athlete_beta * bia_smooth_fm
            method = "athlete"
        else:
            if smooth_window > 1 and non_athlete_idx < len(non_athlete_smooth_fm):
                bia_smooth_fm = non_athlete_smooth_fm[non_athlete_idx]
                non_athlete_idx += 1
            else:
                bia_smooth_fm = bia_fm
            
            cal_fm = non_athlete_alpha + non_athlete_beta * bia_smooth_fm
            method = "non_athlete"
        
        cal_bf_pct = 100.0 * cal_fm / weight if weight > 0 else 0.0
        
        calibrated_data.append({
            'date': bia['date'],
            'weight_lb': weight,
            'bia_percent': bia['bf_pct'],
            'fm_bia': bia_fm,
            'fm_bia_smooth': bia_smooth_fm if smooth_window > 1 else bia_fm,
            'fm_cal': cal_fm,
            'bf_percent_cal': cal_bf_pct,
            'method': method,
            'athlete_mode': bia['athlete_mode']
        })
    
    return calibrated_data

def calculate_metrics(calibrated_data: List[Dict], anchor_pairs: List[Dict]) -> Dict:
    """Calculate calibration metrics."""
    metrics = {}
    
    # Calculate residuals at anchor points
    anchor_residuals = []
    for pair in anchor_pairs:
        # Find corresponding calibrated data
        for cal_data in calibrated_data:
            if abs((cal_data['date'] - pair['bia_date']).total_seconds()) < 3600:  # Within 1 hour
                residual = cal_data['fm_cal'] - pair['dexa_fat_mass']
                anchor_residuals.append(residual)
                break
    
    if anchor_residuals:
        metrics['mae'] = np.mean(np.abs(anchor_residuals))
        metrics['rmse'] = np.sqrt(np.mean(np.array(anchor_residuals)**2))
        metrics['n_anchors'] = len(anchor_residuals)
    else:
        metrics['mae'] = 0.0
        metrics['rmse'] = 0.0
        metrics['n_anchors'] = 0
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Calibrate BIA body fat measurements to DEXA scans')
    parser.add_argument('--bia', required=True, help='BIA CSV file path')
    parser.add_argument('--dxa', required=True, help='DEXA LBM CSV file path')
    parser.add_argument('--weights', required=True, help='Weights CSV file path')
    parser.add_argument('--out', required=True, help='Output CSV file path')
    parser.add_argument('--anchor-window', type=int, default=3, help='Days to search for BIA-DEXA pairs')
    parser.add_argument('--smooth', type=int, default=7, help='Smoothing window in days (0 for none)')
    parser.add_argument('--metrics', help='Output metrics JSON file path')
    
    args = parser.parse_args()
    
    print("Loading data...")
    bia_data = load_bia_data(args.bia)
    dexa_data = load_dexa_data(args.dxa)
    weights_data = load_weights_data(args.weights)
    
    print(f"Loaded {len(bia_data)} BIA measurements, {len(dexa_data)} DEXA scans, {len(weights_data)} weight measurements")
    
    print("Finding anchor pairs...")
    anchor_pairs = find_anchor_pairs(bia_data, dexa_data, weights_data, args.anchor_window)
    print(f"Found {len(anchor_pairs)} anchor pairs")
    
    if len(anchor_pairs) == 0:
        print("Error: No anchor pairs found. Cannot perform calibration.")
        return
    
    print("Calibrating athlete mode...")
    athlete_alpha, athlete_beta = calibrate_athlete_mode(anchor_pairs)
    
    print("Calibrating non-athlete mode...")
    non_athlete_alpha, non_athlete_beta = calibrate_non_athlete_mode(anchor_pairs)
    
    print("Applying calibration...")
    calibrated_data = apply_calibration(bia_data, weights_data, 
                                      athlete_alpha, athlete_beta,
                                      non_athlete_alpha, non_athlete_beta,
                                      args.smooth)
    
    print("Calculating metrics...")
    metrics = calculate_metrics(calibrated_data, anchor_pairs)
    
    print(f"Calibration metrics:")
    print(f"  MAE: {metrics['mae']:.3f} lb")
    print(f"  RMSE: {metrics['rmse']:.3f} lb")
    print(f"  Anchors used: {metrics['n_anchors']}")
    
    # Save calibrated data
    print(f"Saving calibrated data to {args.out}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    with open(args.out, 'w', newline='') as f:
        fieldnames = ['date', 'weight_lb', 'bia_percent', 'fm_bia', 'fm_bia_smooth', 
                     'fm_cal', 'bf_percent_cal', 'method', 'athlete_mode']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for data in calibrated_data:
            row = {k: v for k, v in data.items()}
            row['date'] = data['date'].isoformat()
            writer.writerow(row)
    
    # Save metrics
    if args.metrics:
        with open(args.metrics, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    print("Calibration complete!")

if __name__ == '__main__':
    main()
