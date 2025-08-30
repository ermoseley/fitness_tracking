# 🎯 Weight Tracker Installation Guide

## 🚀 Quick Start Installation

### Option 1: Professional DMG Installer (Recommended)
1. **Download the Installer**
   - Locate `WeightTracker-Installer.dmg` in this folder
   
2. **Open the Installer**
   - Double-click `WeightTracker-Installer.dmg`
   - A window will open showing the installer interface
   
3. **Install to Applications**
   - Drag `WeightTracker.app` from the left side to the `Applications` folder on the right
   - The Applications folder is already linked for your convenience
   
4. **Launch the App**
   - Go to your Applications folder
   - Double-click `WeightTracker` to launch
   - The app will automatically set up your data directory

### Option 2: Direct App Bundle
1. **Locate the App**
   - Find `dist/WeightTracker.app` in this folder
   
2. **Install to Applications**
   - Drag `WeightTracker.app` to your Applications folder
   - Or double-click to run directly from the current location
   
3. **Launch**
   - Double-click `WeightTracker` from Applications

## 📱 What You'll Get

- **Professional App**: Full-featured macOS application
- **Data Management**: Automatic setup of `~/Documents/WeightTracker/` directory
- **CSV Import**: Your existing weight and LBM data will be automatically copied
- **No Dependencies**: Everything is bundled - no Python installation required

## 🔧 First Launch Setup

When you first launch WeightTracker:

1. **Permission Requests**: macOS may ask for permissions to access files
2. **Data Directory**: The app automatically creates `~/Documents/WeightTracker/`
3. **Initial Data**: Your CSV files are copied to the user directory
4. **Ready to Use**: Start tracking your fitness journey immediately!

## 📊 Features Available

- ✅ **Weight Tracking**: Monitor weight over time
- ✅ **Body Composition**: Track lean body mass and body fat
- ✅ **Advanced Analytics**: EMA smoothing, Kalman filtering
- ✅ **Data Visualization**: Generate beautiful charts and plots
- ✅ **Data Export**: Work with CSV files
- ✅ **Trend Analysis**: Identify patterns and progress

## 🛠️ Troubleshooting

### App Won't Launch
- Check that the app is in your Applications folder
- Right-click the app and select "Open" if macOS blocks it
- Ensure you have macOS 10.15 (Catalina) or later

### Data Not Loading
- Verify your CSV files are in `~/Documents/WeightTracker/`
- Check file permissions and accessibility
- Restart the application

### Plots Not Generating
- The app runs in headless mode for stability
- Plots are saved to `~/Documents/WeightTracker/`
- Use the "Open plot" buttons in the GUI to view results

## 📁 File Locations

- **App Bundle**: `/Applications/WeightTracker.app`
- **User Data**: `~/Documents/WeightTracker/`
- **Generated Plots**: `~/Documents/WeightTracker/*.png`
- **CSV Files**: `~/Documents/WeightTracker/*.csv`

## 🎉 You're All Set!

After installation:
1. Launch WeightTracker from Applications
2. Your data will be automatically loaded
3. Start tracking your fitness progress
4. Generate insights with advanced analytics

---

**Need Help?** The app includes comprehensive error handling and will guide you through any issues.

**Enjoy your fitness journey with Weight Tracker!** 🏃‍♂️💪


## Cleaning build artifacts

To remove build outputs and caches (similar to `make clean`):

```bash
./clean.sh
```

To also remove local virtual environments:

```bash
./clean.sh --all
```

## Git hygiene

Ensure build artifacts aren’t committed:
- Keep `dist/`, `build/`, `*.spec`, `installer_temp/`, and `WeightTracker-Installer.dmg` out of git (already covered by `.gitignore`).
- If any of these were previously committed, run:

```bash
git rm -r --cached dist build installer_temp WeightTracker-Installer.dmg *.spec
```

Then commit the removal and push.
