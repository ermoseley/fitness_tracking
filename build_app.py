#!/usr/bin/env python3
"""
Build script for creating Weight Tracker macOS App Bundle
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
    
    # Check if create-dmg is available (for creating installer)
    try:
        subprocess.run(["create-dmg", "--version"], check=True, capture_output=True)
        print("  ✓ create-dmg found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ✗ create-dmg not found. Installing...")
        run_command("brew install create-dmg", "Installing create-dmg")

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
    
    # Create PyInstaller spec file
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets/*', 'assets'),
        ('*.csv', '.'),
        ('kalman.py', '.'),
        ('weight_tracker.py', '.'),
        ('gui.py', '.'),
    ],
    hiddenimports=[
        'numpy',
        'matplotlib',
        'scipy',
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WeightTracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WeightTracker',
)

app = BUNDLE(
    coll,
    name='WeightTracker.app',
    icon='assets/WeightTracker.icns',
    bundle_identifier='com.weighttracker.app',
    info_plist={
        'CFBundleName': 'Weight Tracker',
        'CFBundleDisplayName': 'Weight Tracker',
        'CFBundleVersion': '1.0',
        'CFBundleShortVersionString': '1.0',
        'CFBundlePackageType': 'APPL',
        'CFBundleSignature': '????',
        'CFBundleInfoDictionaryVersion': '6.0',
        'LSMinimumSystemVersion': '10.15',
        'NSHighResolutionCapable': True,
        'LSApplicationCategoryType': 'public.app-category.healthcare-fitness',
        'NSPrincipalClass': 'NSApplication',
    },
)
'''
    
    with open("WeightTracker.spec", "w") as f:
        f.write(spec_content)
    
    print("  ✓ Created PyInstaller spec file")
    
    # Run PyInstaller
    run_command("pyinstaller WeightTracker.spec", "Building with PyInstaller")
    
    # Verify the app bundle was created
    app_path = "dist/WeightTracker.app"
    if not os.path.exists(app_path):
        print("  ✗ App bundle not created!")
        sys.exit(1)
    
    print(f"  ✓ App bundle created at {app_path}")

def create_installer():
    """Create a DMG installer"""
    print("Creating installer...")
    
    # Create a temporary directory for the installer
    installer_dir = "installer_temp"
    if os.path.exists(installer_dir):
        shutil.rmtree(installer_dir)
    os.makedirs(installer_dir)
    
    # Copy the app bundle to the installer directory
    shutil.copytree("dist/WeightTracker.app", f"{installer_dir}/WeightTracker.app")
    
    # Create Applications folder link
    os.symlink("/Applications", f"{installer_dir}/Applications")
    
    # Create DMG
    dmg_name = "WeightTracker-Installer.dmg"
    run_command(f"create-dmg --volname 'Weight Tracker Installer' --window-pos 200 120 --window-size 600 400 --icon-size 100 --icon 'WeightTracker.app' 175 120 --hide-extension 'WeightTracker.app' --app-drop-link 425 120 '{dmg_name}' '{installer_dir}'", "Creating DMG installer")
    
    # Clean up
    shutil.rmtree(installer_dir)
    
    print(f"  ✓ Installer created: {dmg_name}")

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
    
    # Create installer
    create_installer()
    
    print("\n🎉 Build completed successfully!")
    print("\nFiles created:")
    print(f"  • App Bundle: dist/WeightTracker.app")
    print(f"  • Installer: WeightTracker-Installer.dmg")
    print("\nTo install:")
    print("  1. Double-click WeightTracker-Installer.dmg")
    print("  2. Drag WeightTracker.app to the Applications folder")
    print("  3. Double-click WeightTracker.app to run")

if __name__ == "__main__":
    main()
