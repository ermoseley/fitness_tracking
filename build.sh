#!/bin/bash

# Weight Tracker macOS App Bundle Builder
# This script builds a proper .app bundle that can be double-clicked to run

echo "🚀 Building Weight Tracker macOS App Bundle"
echo "=================================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or later and try again"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $python_version detected, but Python $required_version or later is required"
    exit 1
fi

echo "✅ Python $python_version detected"

# Run the build script
echo "Starting build process..."
python3 build_simple.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Build completed successfully!"
    echo ""
    echo "Your Weight Tracker app is ready at: dist/WeightTracker.app"
    echo ""
    echo "To run the app:"
    echo "  1. Double-click dist/WeightTracker.app"
    echo "  2. Or drag it to your Applications folder for permanent installation"
    echo ""
    echo "For detailed installation instructions, see: INSTALL.md"
else
    echo ""
    echo "❌ Build failed. Please check the error messages above."
    exit 1
fi
