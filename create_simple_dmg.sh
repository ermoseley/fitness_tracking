#!/bin/bash
# Create a simple DMG installer for WeightTracker

set -e

echo "Creating simple DMG installer for WeightTracker..."

# Clean up any existing DMG
rm -f WeightTracker-Installer.dmg

# Create a temporary directory
INSTALLER_DIR="installer_temp"
rm -rf "$INSTALLER_DIR"
mkdir -p "$INSTALLER_DIR"

# Copy the app bundle
cp -R "dist/WeightTracker.app" "$INSTALLER_DIR/"

# Create Applications folder link
ln -sf "/Applications" "$INSTALLER_DIR/Applications"

# Create a simple DMG using hdiutil
echo "Creating DMG with hdiutil..."
hdiutil create -volname "BodyMetrics Installer" -srcfolder "$INSTALLER_DIR" -ov -format UDZO "WeightTracker-Installer.dmg"

# Clean up
rm -rf "$INSTALLER_DIR"

echo "✅ DMG installer created: WeightTracker-Installer.dmg"
echo ""
echo "Installation Instructions:"
echo "1. Double-click WeightTracker-Installer.dmg"
echo "2. A window will open showing WeightTracker.app and Applications folder"
echo "3. Drag WeightTracker.app to the Applications folder"
echo "4. Eject the DMG and launch WeightTracker from Applications"
