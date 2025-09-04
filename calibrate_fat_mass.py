#!/usr/bin/env python3
"""
Calibrate impedance-based fat mass measurements using DEXA fat mass data.

This script:
1. Loads DEXA fat mass measurements from data/fat_mass.csv
2. Loads impedance body fat percentages from data/bf_unified.csv
3. Converts impedance body fat % to fat mass using weight data
4. Calibrates impedance fat mass to DEXA fat mass using linear regression
5. Saves calibrated fat mass data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import os

def parse_datetime(date_str):
    """Parse datetime string with multiple format support."""
    formats = [
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse datetime: {date_str}")

def load_dexa_fat_mass():
    """Load DEXA fat mass measurements."""
    print("Loading DEXA fat mass data...")
    
    if not os.path.exists('data/fat_mass.csv'):
        raise FileNotFoundError("DEXA fat mass data not found at data/fat_mass.csv")
    
    df = pd.read_csv('data/fat_mass.csv')
    print(f"Found {len(df)} DEXA fat mass measurements")
    
    dexa_data = []
    for _, row in df.iterrows():
        if pd.isna(row['fat_mass']):
            continue
            
        dt = parse_datetime(row['date'])
        dexa_data.append({
            'date': dt,
            'fat_mass': float(row['fat_mass'])
        })
    
    print(f"Loaded {len(dexa_data)} valid DEXA fat mass measurements")
    for entry in dexa_data:
        print(f"  {entry['date'].strftime('%Y-%m-%d')}: {entry['fat_mass']:.1f} lb")
    
    return dexa_data

def load_weights():
    """Load weight measurements."""
    print("\nLoading weight data...")
    
    if not os.path.exists('data/weights.csv'):
        raise FileNotFoundError("Weight data not found at data/weights.csv")
    
    df = pd.read_csv('data/weights.csv')
    print(f"Found {len(df)} weight measurements")
    
    weights = []
    for _, row in df.iterrows():
        if pd.isna(row['weight']):
            continue
            
        dt = parse_datetime(row['date'])
        weights.append({
            'date': dt,
            'weight': float(row['weight'])
        })
    
    print(f"Loaded {len(weights)} valid weight measurements")
    return weights

def load_impedance_data():
    """Load impedance body fat percentage data."""
    print("\nLoading impedance body fat data...")
    
    if not os.path.exists('data/bf_unified.csv'):
        raise FileNotFoundError("Impedance body fat data not found at data/bf_unified.csv")
    
    df = pd.read_csv('data/bf_unified.csv')
    print(f"Found {len(df)} impedance measurements")
    
    impedance_data = []
    for _, row in df.iterrows():
        if pd.isna(row['body_fat_pct_unified']):
            continue
            
        dt = parse_datetime(row['date'])
        impedance_data.append({
            'date': dt,
            'body_fat_pct': float(row['body_fat_pct_unified'])
        })
    
    print(f"Loaded {len(impedance_data)} valid impedance measurements")
    return impedance_data

def find_closest_weight(weights, target_date, max_hours=24):
    """Find the weight measurement closest to target_date within max_hours."""
    target_dt = target_date
    max_seconds = max_hours * 3600
    
    closest_weight = None
    min_diff = float('inf')
    
    for weight_entry in weights:
        diff_seconds = abs((target_dt - weight_entry['date']).total_seconds())
        if diff_seconds <= max_seconds and diff_seconds < min_diff:
            min_diff = diff_seconds
            closest_weight = weight_entry['weight']
    
    return closest_weight

def convert_bf_to_fat_mass(impedance_data, weights):
    """Convert impedance body fat percentages to fat mass using weight data."""
    print("\nConverting impedance body fat % to fat mass...")
    
    fat_mass_data = []
    conversion_count = 0
    
    for entry in impedance_data:
        # Find closest weight measurement
        closest_weight = find_closest_weight(weights, entry['date'])
        
        if closest_weight is not None:
            # Convert body fat % to fat mass: fat_mass = weight * (body_fat_pct / 100)
            fat_mass = closest_weight * (entry['body_fat_pct'] / 100.0)
            
            fat_mass_data.append({
                'date': entry['date'],
                'body_fat_pct': entry['body_fat_pct'],
                'weight': closest_weight,
                'fat_mass': fat_mass
            })
            conversion_count += 1
    
    print(f"Successfully converted {conversion_count} impedance measurements to fat mass")
    return fat_mass_data

def find_anchor_points(dexa_data, impedance_fat_mass, max_days=7):
    """Find impedance measurements closest to DEXA scan dates."""
    print(f"\nFinding anchor points within {max_days} days of DEXA scans...")
    
    anchor_points = []
    
    for dexa_entry in dexa_data:
        dexa_date = dexa_entry['date']
        dexa_fat_mass = dexa_entry['fat_mass']
        
        closest_impedance = None
        min_diff_days = float('inf')
        
        for imp_entry in impedance_fat_mass:
            diff_days = abs((dexa_date - imp_entry['date']).total_seconds()) / 86400.0
            
            if diff_days <= max_days and diff_days < min_diff_days:
                min_diff_days = diff_days
                closest_impedance = imp_entry
        
        if closest_impedance:
            anchor_points.append({
                'dexa_date': dexa_date,
                'dexa_fat_mass': dexa_fat_mass,
                'impedance_date': closest_impedance['date'],
                'impedance_fat_mass': closest_impedance['fat_mass'],
                'days_diff': min_diff_days
            })
            print(f"  DEXA {dexa_date.strftime('%Y-%m-%d')}: {dexa_fat_mass:.1f} lb")
            print(f"    -> Impedance {closest_impedance['date'].strftime('%Y-%m-%d')}: {closest_impedance['fat_mass']:.1f} lb (diff: {min_diff_days:.1f} days)")
        else:
            print(f"  DEXA {dexa_date.strftime('%Y-%m-%d')}: {dexa_fat_mass:.1f} lb -> No impedance data within {max_days} days")
    
    print(f"Found {len(anchor_points)} anchor points")
    return anchor_points

def calibrate_fat_mass(anchor_points):
    """Calibrate impedance fat mass to DEXA fat mass using linear regression."""
    print("\nCalibrating impedance fat mass to DEXA fat mass...")
    
    if len(anchor_points) < 2:
        raise ValueError("Need at least 2 anchor points for calibration")
    
    # Extract data for regression
    impedance_fat_mass = [ap['impedance_fat_mass'] for ap in anchor_points]
    dexa_fat_mass = [ap['dexa_fat_mass'] for ap in anchor_points]
    
    print(f"Impedance fat mass range: {min(impedance_fat_mass):.1f} - {max(impedance_fat_mass):.1f} lb")
    print(f"DEXA fat mass range: {min(dexa_fat_mass):.1f} - {max(dexa_fat_mass):.1f} lb")
    
    # Linear regression: DEXA_fat_mass = alpha + beta * impedance_fat_mass
    slope, intercept, r_value, p_value, std_err = stats.linregress(impedance_fat_mass, dexa_fat_mass)
    
    print(f"\nCalibration results:")
    print(f"  DEXA_fat_mass = {intercept:.3f} + {slope:.3f} * impedance_fat_mass")
    print(f"  R² = {r_value**2:.4f}")
    print(f"  p-value = {p_value:.4f}")
    print(f"  Standard error = {std_err:.4f}")
    
    # Calculate residuals
    predicted = [intercept + slope * x for x in impedance_fat_mass]
    residuals = [actual - pred for actual, pred in zip(dexa_fat_mass, predicted)]
    rmse = np.sqrt(np.mean([r**2 for r in residuals]))
    
    print(f"  RMSE = {rmse:.3f} lb")
    
    # Show calibration details
    print(f"\nCalibration details:")
    for i, ap in enumerate(anchor_points):
        predicted_fat_mass = intercept + slope * ap['impedance_fat_mass']
        residual = ap['dexa_fat_mass'] - predicted_fat_mass
        print(f"  Point {i+1}: Impedance {ap['impedance_fat_mass']:.1f} -> Predicted {predicted_fat_mass:.1f}, Actual {ap['dexa_fat_mass']:.1f}, Residual {residual:.1f}")
    
    return slope, intercept, r_value**2, rmse

def apply_calibration(impedance_fat_mass, slope, intercept):
    """Apply calibration to all impedance fat mass measurements."""
    print(f"\nApplying calibration to {len(impedance_fat_mass)} impedance measurements...")
    
    calibrated_data = []
    for entry in impedance_fat_mass:
        calibrated_fat_mass = intercept + slope * entry['fat_mass']
        calibrated_body_fat_pct = (calibrated_fat_mass / entry['weight']) * 100.0
        
        calibrated_data.append({
            'date': entry['date'],
            'original_body_fat_pct': entry['body_fat_pct'],
            'original_fat_mass': entry['fat_mass'],
            'calibrated_fat_mass': calibrated_fat_mass,
            'calibrated_body_fat_pct': calibrated_body_fat_pct,
            'weight': entry['weight']
        })
    
    print(f"Calibrated {len(calibrated_data)} measurements")
    return calibrated_data

def save_calibrated_data(calibrated_data, slope, intercept, r_squared, rmse):
    """Save calibrated data to CSV files."""
    print("\nSaving calibrated data...")
    
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save detailed calibrated data
    df = pd.DataFrame(calibrated_data)
    df['date'] = df['date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    df.to_csv('data/calibrated_fat_mass_detailed.csv', index=False)
    print("Saved detailed calibrated data to: data/calibrated_fat_mass_detailed.csv")
    
    # Save simplified calibrated data (just date and calibrated body fat %)
    simple_df = df[['date', 'calibrated_body_fat_pct']].copy()
    simple_df.columns = ['date', 'body_fat_pct_cal']
    simple_df.to_csv('data/calibrated_fat_mass_final.csv', index=False)
    print("Saved final calibrated data to: data/calibrated_fat_mass_final.csv")
    
    # Save calibration parameters
    calibration_params = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'rmse': rmse,
        'calibration_date': datetime.now().isoformat(),
        'equation': f"DEXA_fat_mass = {intercept:.6f} + {slope:.6f} * impedance_fat_mass"
    }
    
    import json
    with open('data/fat_mass_calibration_params.json', 'w') as f:
        json.dump(calibration_params, f, indent=2)
    print("Saved calibration parameters to: data/fat_mass_calibration_params.json")
    
    return df

def main():
    """Main calibration workflow."""
    print("=== Fat Mass Calibration Tool ===")
    print("Calibrating impedance-based fat mass measurements using DEXA data\n")
    
    try:
        # Load data
        dexa_data = load_dexa_fat_mass()
        weights = load_weights()
        impedance_data = load_impedance_data()
        
        # Convert impedance body fat % to fat mass
        impedance_fat_mass = convert_bf_to_fat_mass(impedance_data, weights)
        
        # Find anchor points
        anchor_points = find_anchor_points(dexa_data, impedance_fat_mass)
        
        if len(anchor_points) < 2:
            print(f"ERROR: Need at least 2 anchor points for calibration, found {len(anchor_points)}")
            return
        
        # Calibrate
        slope, intercept, r_squared, rmse = calibrate_fat_mass(anchor_points)
        
        # Apply calibration
        calibrated_data = apply_calibration(impedance_fat_mass, slope, intercept)
        
        # Save results
        df = save_calibrated_data(calibrated_data, slope, intercept, r_squared, rmse)
        
        print(f"\n=== Calibration Complete ===")
        print(f"Calibrated {len(calibrated_data)} impedance measurements")
        print(f"R² = {r_squared:.4f}")
        print(f"RMSE = {rmse:.3f} lb")
        
        # Show summary statistics
        cal_bf_pct = [entry['calibrated_body_fat_pct'] for entry in calibrated_data]
        print(f"\nCalibrated body fat % range: {min(cal_bf_pct):.1f}% - {max(cal_bf_pct):.1f}%")
        print(f"Calibrated body fat % mean: {np.mean(cal_bf_pct):.1f}%")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
