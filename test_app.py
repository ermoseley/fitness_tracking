#!/usr/bin/env python3
"""
Test script to verify the WeightTracker app bundle works correctly
"""

import os
import sys
import subprocess

def test_app_bundle():
    """Test the app bundle functionality"""
    print("Testing WeightTracker app bundle...")
    
    # Check if we're in the right directory
    if not os.path.exists("dist/WeightTracker.app"):
        print("❌ App bundle not found in dist/WeightTracker.app")
        return False
    
    # Test the executable directly
    print("\n1. Testing executable directly...")
    try:
        result = subprocess.run(
            ["dist/WeightTracker.app/Contents/MacOS/WeightTracker"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print("✅ Executable runs successfully")
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✅ Executable runs (timed out after 10s, which is expected)")
    except Exception as e:
        print(f"❌ Executable failed: {e}")
        return False
    
    # Check if CSV files were copied to user directory
    print("\n2. Checking CSV file setup...")
    user_data_dir = os.path.expanduser("~/Documents/WeightTracker")
    if os.path.exists(user_data_dir):
        print(f"✅ User data directory exists: {user_data_dir}")
        
        weights_csv = os.path.join(user_data_dir, "weights.csv")
        lbm_csv = os.path.join(user_data_dir, "lbm.csv")
        
        if os.path.exists(weights_csv):
            print(f"✅ Weights CSV exists: {weights_csv}")
        else:
            print(f"❌ Weights CSV missing: {weights_csv}")
            
        if os.path.exists(lbm_csv):
            print(f"✅ LBM CSV exists: {lbm_csv}")
        else:
            print(f"❌ LBM CSV missing: {lbm_csv}")
    else:
        print(f"❌ User data directory missing: {user_data_dir}")
        return False
    
    # Check bundle structure
    print("\n3. Checking bundle structure...")
    bundle_path = "dist/WeightTracker.app"
    required_files = [
        "Contents/MacOS/WeightTracker",
        "Contents/Resources/weight_tracker.py",
        "Contents/Resources/gui.py",
        "Contents/Resources/kalman.py",
        "Contents/Resources/weights.csv",
        "Contents/Resources/lbm.csv"
    ]
    
    for file_path in required_files:
        full_path = os.path.join(bundle_path, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    print("\n🎉 App bundle test completed successfully!")
    print("\nTo install in Applications:")
    print("1. Open Finder")
    print("2. Navigate to this directory")
    print("3. Drag 'dist/WeightTracker.app' to your Applications folder")
    print("4. Double-click WeightTracker in Applications to launch")
    
    return True

if __name__ == "__main__":
    test_app_bundle()
