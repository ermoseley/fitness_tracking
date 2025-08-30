#!/usr/bin/env python3
"""
Simple build script for WeightTracker
"""

import os
import subprocess
import sys
import shutil

def main():
    print("🚀 Building WeightTracker App Bundle")
    print("=" * 50)
    
    # Check if PyInstaller is available
    try:
        import PyInstaller
        print("✓ PyInstaller found")
    except ImportError:
        print("✗ PyInstaller not found. Install with: pip install pyinstaller")
        return False
    
    # Clean previous build
    print("Cleaning previous build...")
    if os.path.exists("build"):
        shutil.rmtree("build")
        print("✓ Removed build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
        print("✓ Removed dist")
    
    # Create PyInstaller spec content
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('weights.csv', '.'),
        ('lbm.csv', '.'),
        ('assets', 'assets'),
    ],
    hiddenimports=[
        'numpy',
        'matplotlib',
        'scipy',
        'scipy.interpolate',
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
    ],
    hookspath=[],
    hooksconfig={{}},
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
    info_plist={{
        'CFBundleIdentifier': 'com.weighttracker.app',
        'LSMultipleInstancesProhibited': True,
        'NSHighResolutionCapable': True,
    }},
)
'''
    
    # Write spec file
    with open('WeightTracker.spec', 'w') as f:
        f.write(spec_content)
    print("✓ Created PyInstaller spec file")
    
    # Build with PyInstaller
    print("Building with PyInstaller...")
    try:
        result = subprocess.run([
            'pyinstaller', 'WeightTracker.spec', '--clean'
        ], capture_output=True, text=True, check=True)
        print("✓ Building with PyInstaller completed")
    except subprocess.CalledProcessError as e:
        print(f"✗ PyInstaller failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    
    # Check if app was created
    app_path = "dist/WeightTracker.app"
    if os.path.exists(app_path):
        print(f"✓ App bundle created at {app_path}")
    else:
        print("✗ App bundle not found")
        return False
    
    print("\n🎉 Build completed successfully!")
    print(f"App bundle: {app_path}")
    print("\nTo create a DMG installer, run: ./create_simple_dmg.sh")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
