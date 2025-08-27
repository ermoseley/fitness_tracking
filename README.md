# Weight Tracker

A comprehensive weight and body composition tracking application with advanced analytics, trend analysis, and beautiful visualizations.

## 🎉 **NEW: macOS App Bundle Ready!**

Your Weight Tracker is now available as a native macOS `.app` bundle that can be double-clicked to run without opening the terminal!

**Quick Start:**
```bash
# The app bundle is already built and ready!
open dist/WeightTracker.app
```

## Features

- **Weight Tracking**: Log and monitor your weight over time
- **Body Fat Analysis**: Track lean body mass and body fat percentage
- **Advanced Analytics**: Exponential moving averages and regression analysis
- **Kalman Filtering**: Noise reduction for cleaner trend visualization
- **CSV Import/Export**: Easy data management and backup
- **Beautiful Charts**: Professional-grade visualizations using matplotlib
- **Cross-Platform**: Works on macOS, Windows, and Linux
- **Native macOS App**: Professional app bundle with custom icon

## Quick Start

### Option 1: macOS App Bundle (Recommended for Users) ✅
The app bundle is already built and ready to use!

```bash
# Run the app (no installation needed)
open dist/WeightTracker.app

# Or install permanently
# Drag dist/WeightTracker.app to your Applications folder
```

**Features of the App Bundle:**
- 🚀 **Double-click to run** - No terminal required
- 🎨 **Custom icon** - Uses your WeightTracker.icns
- 📱 **Native macOS integration** - Appears in Applications and Launchpad
- 🔒 **Self-contained** - Includes all Python dependencies
- 📦 **Professional packaging** - 384MB complete application

### Option 2: Run from Source (Development)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the GUI application
python gui.py

# Or run the command-line version
python weight_tracker.py --help
```

## Building the macOS App

The build process creates a completely self-contained application that includes all Python dependencies and can run without a terminal.

### Prerequisites
- Python 3.8 or later
- macOS 10.15 (Catalina) or later

### Build Commands

**Simple Build (Recommended):**
```bash
./build.sh
```

**Manual Build:**
```bash
python build_simple.py
```

**Alternative Simple Build:**
```bash
./create_app_bundle.sh
```

### What Gets Built

1. **WeightTracker.app** - A proper macOS app bundle
2. **Self-contained** - Includes all Python dependencies
3. **Native Icon** - Uses your custom WeightTracker.icns icon
4. **No Terminal Required** - Double-click to run

## Installation

### For End Users
1. **Download** the built `WeightTracker.app`
2. **Drag and Drop** to your Applications folder
3. **Launch** from Applications or Spotlight

### For Developers
1. **Clone** the repository
2. **Install** dependencies: `pip install -r requirements.txt`
3. **Build** the app: `./build.sh`
4. **Test** the app bundle

## Data Management

### CSV Files
- **weights.csv**: Weight entries with dates
- **lbm.csv**: Lean body mass entries (optional)

### Data Format
```csv
2025-01-15,190.5
2025-01-16,190.2
2025-01-17,189.8
```

### Importing Data
- Use the GUI to browse and select CSV files
- Supports multiple date formats
- Automatic header detection

## Advanced Features

### Kalman Filtering
- **Smoother Mode**: Post-processing for historical data
- **Filter Mode**: Real-time processing
- **Noise Reduction**: Cleaner trend visualization

### Trend Analysis
- **Exponential Moving Averages**: Smooth trend lines
- **Linear Regression**: Long-term trajectory
- **Statistical Analysis**: Confidence intervals

### Visualization
- **Weight Trends**: Time series with trend lines
- **Body Fat Analysis**: Composition over time
- **Export Options**: High-resolution PNG charts

## Troubleshooting

### Common Issues

**App Won't Launch:**
- Check macOS security settings
- Right-click → Open (first time)
- Verify Python 3.8+ is installed

**Missing Dependencies:**
- Run `pip install -r requirements.txt`
- Ensure all packages are installed

**Build Failures:**
- Check Python version (3.8+ required)
- Verify PyInstaller is installed
- Check available disk space

### Getting Help
- Check the `INSTALL.md` file for detailed instructions
- Review build output for error messages
- Ensure all required files are present

## Development

### Project Structure
```
fitness_tracking/
├── main.py              # App entry point
├── gui.py               # GUI implementation
├── weight_tracker.py    # Core functionality
├── kalman.py           # Kalman filtering
├── build_simple.py     # Build script
├── build.sh            # Build shell script
├── create_app_bundle.sh # Simple app bundle creator
├── assets/             # Icons and resources
├── *.csv               # Data files
├── dist/               # Built app bundle (ready to use!)
└── requirements.txt    # Dependencies
```

### Adding Features
1. **Modify** the appropriate Python module
2. **Test** with `python gui.py`
3. **Rebuild** the app: `./build.sh`
4. **Test** the new app bundle

### Customization
- **Icons**: Replace files in `assets/`
- **Styling**: Modify GUI elements in `gui.py`
- **Analysis**: Extend algorithms in `weight_tracker.py`

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Contributing

Contributions are welcome! Please:
1. Test your changes thoroughly
2. Ensure the app bundle still builds correctly
3. Update documentation as needed
4. Follow existing code style

---

**🎉 Your Weight Tracker is now a professional macOS application!**

**Built with ❤️ using Python, Tkinter, and PyInstaller**
