# Weight Tracker

A comprehensive weight and body composition tracking application with advanced analytics, trend analysis, and data visualization.

## Overview

Weight Tracker is now available as a native macOS application bundle that can be launched by double-clicking, eliminating the need for terminal usage.

## Features

- Weight tracking with historical data logging
- Body fat analysis with lean body mass calculations
- Advanced analytics including exponential moving averages and regression analysis
- Kalman filtering for signal processing and noise reduction
- CSV import/export functionality for data management
- Chart visualization using matplotlib
- Cross-platform compatibility (macOS, Windows, Linux)
- Native macOS application bundle with custom icon

## Quick Start

### Using the macOS Application Bundle

The application bundle is ready to use immediately:

```bash
# Launch the application
open dist/WeightTracker.app
```

For permanent installation, drag `dist/WeightTracker.app` to your Applications folder.

#### Application Bundle Characteristics:
- Double-click execution without terminal dependency
- Custom icon integration
- Native macOS integration with Applications and Launchpad
- Self-contained package with all Python dependencies included
- Complete application package (approximately 384MB)

### Running from Source Code
```bash
# Install required dependencies
pip install -r requirements.txt

# Launch the graphical user interface
python gui.py

# Or use the command-line interface
python weight_tracker.py --help
```

## Building the macOS Application

The build process creates a self-contained application package that includes all Python dependencies and can be executed without terminal intervention.

### Prerequisites
- Python 3.8 or later
- macOS 10.15 (Catalina) or later

### Build Commands

#### Primary Build Method (Recommended):
```bash
./build.sh
```

#### Alternative Build Methods:
```bash
python build_simple.py
# or
./create_app_bundle.sh
```

### Build Output

The build process generates:
1. WeightTracker.app - A native macOS application bundle
2. Self-contained package with all Python dependencies included
3. Custom icon integration using WeightTracker.icns
4. Double-click execution capability without terminal dependency

## Installation

### For End Users
1. Obtain the built WeightTracker.app file
2. Drag the application to your Applications folder
3. Launch from Applications or Spotlight

### For Developers
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Execute build script: `./build.sh`
4. Test the generated application bundle

## Data Management

### CSV Files
- weights.csv: Weight entries with associated dates
- lbm.csv: Lean body mass entries (optional)

### Data Format
```csv
2025-01-15,190.5
2025-01-16,190.2
2025-01-17,189.8
```

### Importing Data
- Use the graphical interface to browse and select CSV files
- Supports multiple date format specifications
- Automatic header detection and parsing

## Advanced Features

### Kalman Filtering
- Smoother mode for post-processing historical data
- Filter mode for real-time processing
- Signal processing with noise reduction for improved visualization

### Trend Analysis
- Exponential moving averages for trend smoothing
- Linear regression for trajectory analysis
- Statistical confidence interval calculations

### Visualization
- Weight trend time series with trend line overlays
- Body fat composition analysis over time
- High-resolution PNG chart export capabilities

## Troubleshooting

### Common Issues

#### Application Will Not Launch:
- Verify macOS security settings in System Preferences
- Use right-click → Open for first-time execution
- Ensure Python 3.8 or later is installed

#### Missing Dependencies:
- Execute: `pip install -r requirements.txt`
- Verify all required packages are installed

#### Build Failures:
- Confirm Python version 3.8 or later
- Ensure PyInstaller is installed
- Check available disk space

#### Support Resources:
- Refer to INSTALL.md for detailed setup instructions
- Examine build output for specific error messages
- Verify all required files are present in the project

## Development

### Project Structure
```
fitness_tracking/
├── main.py              # Application entry point
├── gui.py               # Graphical user interface implementation
├── weight_tracker.py    # Core functionality and algorithms
├── kalman.py           # Kalman filtering implementation
├── build_simple.py     # Build automation script
├── build.sh            # Primary build shell script
├── create_app_bundle.sh # Alternative build script
├── assets/             # Icons and resource files
├── *.csv               # Data storage files
├── dist/               # Generated application bundle
└── requirements.txt    # Python dependencies
```

### Adding Features
1. Modify the appropriate Python module
2. Test using: `python gui.py`
3. Rebuild with: `./build.sh`
4. Validate the new application bundle

### Customization Options
- Icons: Replace files in the assets/ directory
- Interface styling: Modify GUI elements in gui.py
- Algorithms: Extend functionality in weight_tracker.py

## License

This project is open source. Modify and distribute according to your requirements.

## Contributing

Contributions are welcome. Please ensure:
1. Thorough testing of all changes
2. Application bundle builds successfully
3. Documentation is updated as needed
4. Existing code style conventions are followed
