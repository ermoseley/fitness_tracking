# BodyMetrics - Streamlit Version

A modern web-based fitness tracking application built with Streamlit, featuring advanced Kalman filtering for weight trend analysis and body composition tracking.

## Features

### 🔐 Authentication
- **User Registration & Login**: Create accounts with username and password
- **Secure Password Storage**: Passwords are hashed using PBKDF2
- **Session Management**: 24-hour session timeout for security
- **Data Isolation**: Each user's data is completely separate

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

1. **First Time Setup**: 
   - Register a new account with a username and password
   - Upload your weight data CSV file in the Data Management page
2. **Daily Use**: 
   - Login with your credentials
   - Add new weight entries using the Add Entries page
   - View your progress on the Dashboard
3. **Analysis**: 
   - Check the Weight Analysis page for detailed trends
   - Set your height and upload LBM data for body fat analysis
   - Customize settings in the Settings page
4. **Data Management**: 
   - Export your data as CSV files
   - Upload new data files to replace existing data

## Technical Details

- **Kalman Filtering**: Advanced state estimation for smooth weight trends
- **Time-aware EMA**: Exponential moving average that accounts for irregular sampling
- **Interactive Visualizations**: Built with Plotly for responsive charts
- **Data Persistence**:
  - Local dev: stored in SQLite at `data/fitness_tracker.db`
  - Cloud: optionally use Postgres via `DATABASE_URL` (e.g., Supabase). Set in Streamlit secrets or environment.

### Cloud Deployment (Streamlit Cloud)

1. Push this repo to GitHub (public or private).
2. In Streamlit Cloud, deploy the app pointing to `streamlit_app.py`.
3. Configure secrets (if using managed Postgres):

```toml
[general]
email = "you@example.com"

[connections]

[server]

# Application secrets
DATABASE_URL = "postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME"
```

4. If you do not set `DATABASE_URL`, the app will fall back to SQLite and store data in the app's filesystem (may be ephemeral on some platforms).

## Comparison with Original GUI

This Streamlit version provides:
- Modern web-based interface
- Better data visualization with interactive charts
- Improved data management capabilities
- Mobile-friendly responsive design
- No need for desktop application installation

The core analysis algorithms (Kalman filtering, EMA, regression) remain identical to the original implementation.
