# Weight Tracker - Streamlit Version

A modern web-based fitness tracking application built with Streamlit, featuring advanced Kalman filtering for weight trend analysis and body composition tracking.

## Features

### 🏠 Dashboard
- Overview of current weight and trends
- Key metrics including 7-day EMA and trend rates
- Interactive weight trend visualization
- Recent entries table

### 📈 Weight Analysis
- Add new weight entries with date/time
- Kalman filter analysis with confidence intervals
- Advanced trend analysis and forecasting
- Interactive charts with Plotly

### 📊 Composition Analysis
- BMI calculation and trend analysis
- Body fat percentage tracking (requires LBM data)
- Height management
- Category-based health indicators

### ⚙️ Settings
- Configurable Kalman filter parameters
- Plot customization options
- Data export functionality
- Confidence interval settings

### 📁 Data Management
- Upload CSV files for weights and LBM data
- Data preview and summary
- Export functionality

## Installation

1. Install dependencies:
```bash
pip install -r requirements_streamlit.txt
```

2. Run the application:
```bash
streamlit run streamlit_app.py
```

3. Open your browser to the URL shown in the terminal (typically http://localhost:8501)

## Data Format

### Weights CSV
Should contain columns:
- `date`: Date/time in ISO format or other supported formats
- `weight`: Weight in pounds

### LBM CSV (optional)
Should contain columns:
- `date`: Date in ISO format
- `lbm`: Lean body mass in pounds

## Usage

1. **First Time Setup**: Upload your weight data CSV file in the Data Management page
2. **Add Entries**: Use the Weight Tracking page to add new weight entries
3. **View Analysis**: Check the Dashboard for overview metrics and trends
4. **Body Composition**: Set your height and upload LBM data for body fat analysis
5. **Customize**: Adjust settings in the Settings page for different analysis parameters

## Technical Details

- **Kalman Filtering**: Advanced state estimation for smooth weight trends
- **Time-aware EMA**: Exponential moving average that accounts for irregular sampling
- **Interactive Visualizations**: Built with Plotly for responsive charts
- **Data Persistence**: All data stored in local CSV files in the `data/` directory

## Comparison with Original GUI

This Streamlit version provides:
- Modern web-based interface
- Better data visualization with interactive charts
- Improved data management capabilities
- Mobile-friendly responsive design
- No need for desktop application installation

The core analysis algorithms (Kalman filtering, EMA, regression) remain identical to the original implementation.
