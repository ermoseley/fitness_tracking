#!/bin/bash

# Simple Weight Tracker App Bundle Creator
# This script creates a basic .app bundle that can run the Python script

echo "🚀 Creating Weight Tracker App Bundle"
echo "====================================="

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/WeightTracker.app
rm -rf build

# Create app bundle structure
echo "Creating app bundle structure..."
mkdir -p dist/WeightTracker.app/Contents/MacOS
mkdir -p dist/WeightTracker.app/Contents/Resources

# Create the launcher script
cat > dist/WeightTracker.app/Contents/MacOS/WeightTracker << 'EOF'
#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESOURCES_DIR="$APP_DIR/Contents/Resources"

# Change to the resources directory
cd "$RESOURCES_DIR"

# Run the Python script
python3 main.py "$@"
EOF

# Make the launcher executable
chmod +x dist/WeightTracker.app/Contents/MacOS/WeightTracker

# Create Info.plist
cat > dist/WeightTracker.app/Contents/Info.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
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
</plist>
EOF

# Copy all necessary files to Resources
echo "Copying application files..."
cp -r assets dist/WeightTracker.app/Contents/Resources/
cp *.py dist/WeightTracker.app/Contents/Resources/
cp *.csv dist/WeightTracker.app/Contents/Resources/ 2>/dev/null || true

# Copy icon if available
if [ -f "assets/WeightTracker.icns" ]; then
    cp assets/WeightTracker.icns dist/WeightTracker.app/Contents/Resources/
    # Update Info.plist to include icon
    sed -i '' 's/<key>NSPrincipalClass<\/key>/<key>CFBundleIconFile<\/key>\n    <string>WeightTracker.icns<\/string>\n    <key>NSPrincipalClass<\/key>/' dist/WeightTracker.app/Contents/Info.plist
fi

# Create a simple requirements.txt in Resources
cat > dist/WeightTracker.app/Contents/Resources/requirements.txt << 'EOF'
numpy>=1.24
matplotlib>=3.8
scipy>=1.11
EOF

# Create installation instructions
cat > INSTALL.md << 'EOF'
# Weight Tracker Installation Instructions

## Quick Installation
1. **Double-click** `WeightTracker.app` in the `dist` folder to run
2. **Drag and Drop** to your Applications folder for permanent installation

## Requirements
- Python 3.8 or later must be installed on your system
- Required packages: `pip install -r requirements.txt`

## First Run
- macOS may ask for permission to run the app
- Right-click → Open if you get a security warning

## Features
- Weight tracking with CSV import/export
- Body fat percentage tracking
- Trend analysis with exponential moving averages
- Kalman filtering for noise reduction
- Beautiful charts and visualizations

## Troubleshooting
- Ensure Python 3.8+ is installed and in PATH
- Install required packages: `pip install numpy matplotlib scipy`
- Check that all files are present in the app bundle
EOF

echo ""
echo "🎉 App bundle created successfully!"
echo ""
echo "Your Weight Tracker app is ready at: dist/WeightTracker.app"
echo ""
echo "To run the app:"
echo "  1. Double-click dist/WeightTracker.app"
echo "  2. Or drag it to your Applications folder"
echo ""
echo "Note: This app bundle requires Python 3.8+ and the required packages to be installed on your system."
echo "For a completely self-contained app, use the PyInstaller build script instead."
echo ""
echo "Installation guide created: INSTALL.md"
