#!/usr/bin/env python3
"""
Simple build script for creating Weight Tracker macOS App Bundle
This script creates a proper .app bundle that can be double-clicked to run.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"  {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"    ✓ {description} completed")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"    ✗ {description} failed: {e}")
        print(f"    Error output: {e.stderr}")
        sys.exit(1)

def check_dependencies():
    """Check if required tools are available"""
    print("Checking dependencies...")
    
    # Check if pyinstaller is available
    try:
        subprocess.run(["pyinstaller", "--version"], check=True, capture_output=True)
        print("  ✓ PyInstaller found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ✗ PyInstaller not found. Installing...")
        run_command("pip install pyinstaller", "Installing PyInstaller")

def clean_build():
    """Clean previous build artifacts"""
    print("Cleaning previous build...")
    
    dirs_to_clean = ["build", "dist", "__pycache__"]
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"  ✓ Removed {dir_name}")
    
    # Clean .spec files
    for spec_file in Path(".").glob("*.spec"):
        spec_file.unlink()
        print(f"  ✓ Removed {spec_file}")

def build_app_bundle():
    """Build the macOS app bundle using PyInstaller"""
    print("Building app bundle...")
    
    # Use PyInstaller's --onefile --windowed options for simplicity
    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name=WeightTracker",
        "--icon=assets/WeightTracker.icns",
        "--add-data=assets:assets",
        "--add-data=*.csv:.",
        "--add-data=kalman.py:.",
        "--add-data=weight_tracker.py:.",
        "--add-data=gui.py:.",
        "--hidden-import=numpy",
        "--hidden-import=matplotlib",
        "--hidden-import=scipy",
        "--hidden-import=tkinter",
        "--hidden-import=tkinter.filedialog",
        "--hidden-import=tkinter.messagebox",
        "--hidden-import=tkinter.scrolledtext",
        "main.py"
    ]
    
    # Run PyInstaller
    run_command(" ".join(cmd), "Building with PyInstaller")
    
    # Verify the executable was created
    exe_path = "dist/WeightTracker"
    if not os.path.exists(exe_path):
        print("  ✗ Executable not created!")
        sys.exit(1)
    
    print(f"  ✓ Executable created at {exe_path}")
    
    # Create a simple app bundle structure
    create_simple_app_bundle(exe_path)

def create_simple_app_bundle(exe_path):
    """Create a simple app bundle structure"""
    print("Creating app bundle structure...")
    
    app_name = "WeightTracker.app"
    app_path = f"dist/{app_name}"
    contents_path = f"{app_path}/Contents"
    macos_path = f"{contents_path}/MacOS"
    
    # Create directory structure
    os.makedirs(macos_path, exist_ok=True)
    
    # Copy the executable
    shutil.copy2(exe_path, f"{macos_path}/WeightTracker")
    
    # Create Info.plist
    info_plist_content = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>WeightTracker</string>
    <key>CFBundleIdentifier</key>
    <string>com.weighttracker.app</string>
    <key>CFBundleName</key>
    <string>Weight Tracker</string>
    <key>CFBundleDisplayName</key>
    <string>Weight Tracker</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.healthcare-fitness</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
</dict>
</plist>'''
    
    with open(f"{contents_path}/Info.plist", "w") as f:
        f.write(info_plist_content)
    
    # Copy icon if available
    icon_source = "assets/WeightTracker.icns"
    if os.path.exists(icon_source):
        shutil.copy2(icon_source, f"{contents_path}/WeightTracker.icns")
        # Update Info.plist to include icon
        info_plist_content = info_plist_content.replace(
            '<key>NSPrincipalClass</key>',
            '<key>CFBundleIconFile</key>\n    <string>WeightTracker.icns</string>\n    <key>NSPrincipalClass</key>'
        )
        with open(f"{contents_path}/Info.plist", "w") as f:
            f.write(info_plist_content)
    
    # Make the executable executable
    os.chmod(f"{macos_path}/WeightTracker", 0o755)
    
    print(f"  ✓ App bundle created at {app_path}")

def create_install_instructions():
    """Create installation instructions"""
    print("Creating installation instructions...")
    
    instructions = """# Weight Tracker Installation Instructions

## Option 1: Direct Installation (Recommended)
1. Double-click `WeightTracker.app` in the `dist` folder
2. The app will run directly from its current location
3. To install permanently, drag `WeightTracker.app` to your Applications folder

## Option 2: Install to Applications
1. Open Finder
2. Navigate to the `dist` folder
3. Drag `WeightTracker.app` to your Applications folder
4. The app will now appear in your Applications and Launchpad

## Option 3: Create a DMG Installer (Advanced)
If you want to create a proper DMG installer like commercial apps:

1. Install create-dmg: `brew install create-dmg`
2. Run the full build script: `python build_app.py`

## Troubleshooting
- If the app doesn't run, make sure you have Python 3.8+ installed
- The app includes all necessary dependencies and doesn't require additional Python packages
- On first run, macOS may ask for permission to run the app

## App Features
- Weight tracking with CSV import/export
- Body fat percentage tracking
- Trend analysis with exponential moving averages
- Kalman filtering for noise reduction
- Beautiful charts and visualizations
"""
    
    with open("INSTALL.md", "w") as f:
        f.write(instructions)
    
    print("  ✓ Created installation instructions")

def main():
    """Main build process"""
    print("🚀 Building Weight Tracker macOS App Bundle")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Clean previous builds
    clean_build()
    
    # Build the app bundle
    build_app_bundle()
    
    # Create installation instructions
    create_install_instructions()
    
    print("\n🎉 Build completed successfully!")
    print("\nFiles created:")
    print(f"  • App Bundle: dist/WeightTracker.app")
    print(f"  • Installation Guide: INSTALL.md")
    print("\nTo run the app:")
    print("  1. Double-click dist/WeightTracker.app")
    print("  2. Or drag it to your Applications folder for permanent installation")
    print("\nThe app bundle is completely self-contained and includes all dependencies!")

if __name__ == "__main__":
    main()
