#!/usr/bin/env python3
"""
Unified Body Fat Calibration Tool

Calibrates unified body fat percentages to DEXA body fat percentages
using linear mappings on fat mass (lb), not percent.

Usage:
    python calibrate_unified_bf.py --bia data/bf_unified.csv --dxa data/bf.csv --weights data/weights.csv --out data/calibrated_unified_bf.csv
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
    """Load unified BIA data from CSV file."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                date = parse_datetime(row['date'])
                bf_pct = float(row['body_fat_pct_unified'])
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
    """Load DEXA body fat percentage data."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if not row['date'] or not row['bf']:  # Skip empty rows
                    continue
                date = parse_datetime(row['date'])
                bf_pct = float(row['bf'])
                data.append({
                    'date': date,
                    'bf_pct': bf_pct
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
        dexa_bf_pct = dexa['bf_pct']
        
        # Get weight for DEXA date to calculate fat mass
        dexa_weight = interpolate_weight(dexa_date, weights_data)
        if dexa_weight is None:
            print(f"Warning: No weight data found for DEXA date {dexa_date}")
            continue
        
        # Calculate DEXA fat mass
        dexa_fat_mass = dexa_weight * dexa_bf_pct / 100.0
        
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
                'dexa_bf_pct': dexa_bf_pct,
                'bia_date': best_bia['date'],
                'bia_fat_mass': bia_fat_mass,
                'bia_bf_pct': best_bia['bf_pct'],
                'bia_athlete_mode': best_bia['athlete_mode'],
                'weight': bia_weight,
                'time_diff_days': min_diff
            })
            print(f"Anchor pair: DEXA {dexa_date.date()} (BF%={dexa_bf_pct:.1f}%, FM={dexa_fat_mass:.2f}lb) <-> BIA {best_bia['date'].date()} (BF%={best_bia['bf_pct']:.1f}%, FM={bia_fat_mass:.2f}lb)")
        else:
            print(f"Warning: No BIA measurement found within {anchor_window_days} days of DEXA {dexa_date.date()}")
    
    return anchor_pairs

def calibrate_global_ols(anchor_pairs: List[Dict]) -> Tuple[float, float]:
    """Calibrate using global OLS regression on all anchor pairs."""
    if len(anchor_pairs) < 1:
        print("Warning: No anchor pairs found, using identity calibration")
        return 0.0, 1.0
    
    bia_fm = np.array([p['bia_fat_mass'] for p in anchor_pairs])
    dexa_fm = np.array([p['dexa_fat_mass'] for p in anchor_pairs])
    
    if len(anchor_pairs) == 1:
        # Baseline offset method
        alpha = dexa_fm[0] - bia_fm[0]
        beta = 1.0
        print(f"Calibration (baseline offset): α={alpha:.3f}, β={beta:.3f}")
    else:
        # OLS regression
        beta, alpha, r_value, p_value, std_err = stats.linregress(bia_fm, dexa_fm)
        print(f"Calibration (OLS): α={alpha:.3f}, β={beta:.3f}, R²={r_value**2:.3f}")
    
    return alpha, beta

def calibrate_two_point(anchor_pairs: List[Dict]) -> Tuple[float, float]:
    """Calibrate using exact two-point method for perfect anchor matches."""
    if len(anchor_pairs) < 2:
        print("Warning: Need at least 2 anchor pairs for two-point calibration, using OLS")
        return calibrate_global_ols(anchor_pairs)
    
    # Use first two anchor pairs for exact line
    p1, p2 = anchor_pairs[0], anchor_pairs[1]
    x1, y1 = p1['bia_fat_mass'], p1['dexa_fat_mass']
    x2, y2 = p2['bia_fat_mass'], p2['dexa_fat_mass']
    
    if x2 != x1:
        beta = (y2 - y1) / (x2 - x1)
        alpha = y1 - beta * x1
    else:
        beta = 1.0
        alpha = y1 - x1
    
    print(f"Calibration (two-point): α={alpha:.6f}, β={beta:.6f}")
    
    # Verify exact matches
    for i, pair in enumerate(anchor_pairs[:2]):
        bia_fm = pair['bia_fat_mass']
        cal_fm = alpha + beta * bia_fm
        cal_bf_pct = 100.0 * cal_fm / pair['weight']
        print(f"  Anchor {i+1}: BIA {pair['bia_date'].date()} {pair['bia_bf_pct']:.1f}% -> {cal_bf_pct:.1f}% (DEXA: {pair['dexa_bf_pct']:.1f}%)")
    
    return alpha, beta

def calibrate_piecewise(anchor_pairs: List[Dict]) -> List[Dict]:
    """Calibrate using piecewise linear segments between consecutive DEXA points."""
    if len(anchor_pairs) < 2:
        # Fall back to global OLS if not enough points
        alpha, beta = calibrate_global_ols(anchor_pairs)
        return [{'alpha': alpha, 'beta': beta, 'start_date': None, 'end_date': None}]
    
    segments = []
    sorted_pairs = sorted(anchor_pairs, key=lambda x: x['dexa_date'])
    
    for i in range(len(sorted_pairs) - 1):
        pair1 = sorted_pairs[i]
        pair2 = sorted_pairs[i + 1]
        
        # Calculate segment parameters
        x1, y1 = pair1['bia_fat_mass'], pair1['dexa_fat_mass']
        x2, y2 = pair2['bia_fat_mass'], pair2['dexa_fat_mass']
        
        if x2 != x1:
            beta = (y2 - y1) / (x2 - x1)
            alpha = y1 - beta * x1
        else:
            beta = 1.0
            alpha = y1 - x1
        
        segments.append({
            'alpha': alpha,
            'beta': beta,
            'start_date': pair1['dexa_date'],
            'end_date': pair2['dexa_date']
        })
        print(f"Segment {i+1}: α={alpha:.3f}, β={beta:.3f} (from {pair1['dexa_date'].date()} to {pair2['dexa_date'].date()})")
    
    return segments

def apply_calibration(bia_data: List[Dict], weights_data: List[Dict], 
                     calibration_params: List[Dict], method: str,
                     smooth_window: int = 7) -> List[Dict]:
    """Apply calibration to all BIA data."""
    calibrated_data = []
    
    # Smooth fat mass if requested
    if smooth_window > 1:
        bia_fm = [interpolate_weight(d['date'], weights_data) * d['bf_pct'] / 100.0 
                 for d in bia_data if interpolate_weight(d['date'], weights_data) is not None]
        bia_smooth_fm = moving_median(bia_fm, smooth_window) if bia_fm else []
    else:
        bia_smooth_fm = []
    
    smooth_idx = 0
    
    for bia in bia_data:
        weight = interpolate_weight(bia['date'], weights_data)
        if weight is None:
            continue
        
        bia_fm = weight * bia['bf_pct'] / 100.0
        
        # Get smoothed fat mass if available
        if smooth_window > 1 and smooth_idx < len(bia_smooth_fm):
            bia_smooth_fm_val = bia_smooth_fm[smooth_idx]
            smooth_idx += 1
        else:
            bia_smooth_fm_val = bia_fm
        
        # Apply appropriate calibration based on method
        if method == "piecewise":
            # Find appropriate segment
            cal_alpha, cal_beta = 0.0, 1.0
            segment_id = 0
            
            for i, segment in enumerate(calibration_params):
                if segment['start_date'] is None or segment['end_date'] is None:
                    # Global segment
                    cal_alpha, cal_beta = segment['alpha'], segment['beta']
                    segment_id = i
                elif segment['start_date'] <= bia['date'] <= segment['end_date']:
                    cal_alpha, cal_beta = segment['alpha'], segment['beta']
                    segment_id = i
                    break
                elif bia['date'] < segment['start_date']:
                    # Use this segment for extrapolation
                    cal_alpha, cal_beta = segment['alpha'], segment['beta']
                    segment_id = i
                    break
                elif i == len(calibration_params) - 1:
                    # Use last segment for extrapolation
                    cal_alpha, cal_beta = segment['alpha'], segment['beta']
                    segment_id = i
        else:
            # Global calibration
            cal_alpha, cal_beta = calibration_params[0]['alpha'], calibration_params[0]['beta']
            segment_id = 0
        
        cal_fm = cal_alpha + cal_beta * bia_smooth_fm_val
        cal_bf_pct = 100.0 * cal_fm / weight if weight > 0 else 0.0
        
        calibrated_data.append({
            'date': bia['date'],
            'weight_lb': weight,
            'bia_percent': bia['bf_pct'],
            'fm_bia': bia_fm,
            'fm_bia_smooth': bia_smooth_fm_val,
            'fm_cal': cal_fm,
            'bf_percent_cal': cal_bf_pct,
            'method': method,
            'segment_id': segment_id,
            'alpha': cal_alpha,
            'beta': cal_beta,
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
    parser = argparse.ArgumentParser(description='Calibrate unified body fat measurements to DEXA scans')
    parser.add_argument('--bia', required=True, help='Unified BIA CSV file path')
    parser.add_argument('--dxa', required=True, help='DEXA BF CSV file path')
    parser.add_argument('--weights', required=True, help='Weights CSV file path')
    parser.add_argument('--out', required=True, help='Output CSV file path')
    parser.add_argument('--method', choices=['baseline_offset', 'two_point', 'global_ols', 'piecewise'], 
                       default='global_ols', help='Calibration method')
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
    
    print(f"Calibrating using {args.method} method...")
    if args.method == "piecewise":
        calibration_params = calibrate_piecewise(anchor_pairs)
    elif args.method == "two_point":
        alpha, beta = calibrate_two_point(anchor_pairs)
        calibration_params = [{'alpha': alpha, 'beta': beta, 'start_date': None, 'end_date': None}]
    else:
        alpha, beta = calibrate_global_ols(anchor_pairs)
        calibration_params = [{'alpha': alpha, 'beta': beta, 'start_date': None, 'end_date': None}]
    
    print("Applying calibration...")
    calibrated_data = apply_calibration(bia_data, weights_data, calibration_params, 
                                      args.method, args.smooth)
    
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
                     'fm_cal', 'bf_percent_cal', 'method', 'segment_id', 'alpha', 'beta', 'athlete_mode']
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
