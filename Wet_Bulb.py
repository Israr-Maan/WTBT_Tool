# Enhanced Interactive Wet Bulb Temperature Dashboard
# Streamlit Application with Advanced Features

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Enhanced Wet Bulb Temperature Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Station configuration - FIXED: Updated URLs to use raw GitHub links
GITHUB_BASE_URL = "https://raw.githubusercontent.com/Israr-Maan/WTBT_Tool/master/"
STATIONS = {
    "Station_01_Thatta": "thatta_with_wetbulb.csv",
    "Station_02_Tando_Jam": "tando_jam_with_wetbulb.csv",
    "Station_03_Sukkur": "sukkur_with_wetbulb.csv",
    "Station_04_Sakrand": "sakrand_with_wetbulb.csv",
    "Station_05_Rohri": "rohri_with_wetbulb.csv",
    "Station_06_Padidan": "padidan_with_wetbulb.csv",
    "Station_07_Nawabshah": "nawabshah_with_wetbulb.csv",
    "Station_08_Moen_Jo_Daro": "moen_jo_daro_with_wetbulb.csv",
    "Station_09_Mithi": "mithi_with_wetbulb.csv",
    "Station_10_Mirpur_Khas": "mirpur_khas_with_wetbulb.csv",
    "Station_11_Larkana": "larkana_with_wetbulb.csv",
    "Station_12_Kiamari": "kiamari_with_wetbulb.csv",  # FIXED: Changed from "Liamari" to "Kiamari"
    "Station_13_Jacobabad": "jacobabad_with_wetbulb.csv",
    "Station_14_Hyderabad": "hyderabad_with_wetbulb.csv"
}

# Custom themes
THEMES = {
    "Default": {
        "primary_color": "#FF6B6B",
        "secondary_color": "#4ECDC4",
        "accent_color": "#45B7D1",
        "background_color": "#FFFFFF",
        "text_color": "#2C3E50"
    },
    "Dark Mode": {
        "primary_color": "#E74C3C",
        "secondary_color": "#1ABC9C",
        "accent_color": "#3498DB",
        "background_color": "#2C3E50",
        "text_color": "#ECF0F1"
    },
    "Ocean Blue": {
        "primary_color": "#3498DB",
        "secondary_color": "#2ECC71",
        "accent_color": "#F39C12",
        "background_color": "#EBF5FB",
        "text_color": "#1B4F72"
    },
    "Sunset": {
        "primary_color": "#E67E22",
        "secondary_color": "#E74C3C",
        "accent_color": "#F1C40F",
        "background_color": "#FEF9E7",
        "text_color": "#784212"
    },
    "Forest": {
        "primary_color": "#27AE60",
        "secondary_color": "#2ECC71",
        "accent_color": "#F39C12",
        "background_color": "#E8F8F5",
        "text_color": "#0E4B22"
    }
}

# Custom CSS for styling with theme support
def apply_custom_css(theme):
    st.markdown(f"""
    <style>
        .main-header {{
            font-size: 3rem;
            color: {theme['text_color']};
            text-align: center;
            font-weight: bold;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-container {{
            background: linear-gradient(135deg, {theme['background_color']} 0%, {theme['primary_color']}15 100%);
            padding: 1rem;
            border-radius: 15px;
            margin: 0.5rem 0;
            border: 1px solid {theme['primary_color']}30;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .sidebar .sidebar-content {{
            background: linear-gradient(180deg, {theme['background_color']} 0%, {theme['secondary_color']}10 100%);
        }}
        .stAlert {{
            margin-top: 1rem;
            border-radius: 10px;
        }}
        .chart-container {{
            background-color: {theme['background_color']};
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }}
        .stSelectbox > div > div {{
            background-color: {theme['background_color']};
            border: 2px solid {theme['primary_color']};
        }}
    </style>
    """, unsafe_allow_html=True)

# Load data from GitHub - FIXED: Added better error handling
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_station_data(station_file):
    """Load data from GitHub repository"""
    try:
        url = GITHUB_BASE_URL + station_file
        response = requests.get(url, timeout=30)  # Added timeout
        response.raise_for_status()
        
        # Read CSV from string content
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        # Basic validation
        if df.empty:
            return None, "Empty dataset"
        
        # Check for required columns
        required_columns = ['time', 'Temp', 'Humidity', 'WetBulb']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return None, f"Missing required columns: {missing_columns}"
        
        return df, None
    except requests.RequestException as e:
        return None, f"Network error: {str(e)}"
    except pd.errors.EmptyDataError:
        return None, "Empty or corrupted CSV file"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# Enhanced data processing - FIXED: Added better error handling and null checks
@st.cache_data
def process_station_data(df, station_name):
    """Enhanced data preprocessing with additional features"""
    try:
        df = df.copy()
        
        # Convert time to datetime with error handling
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        
        # Remove rows with invalid timestamps
        df = df.dropna(subset=['time'])
        
        if df.empty:
            return df
        
        df = df.sort_values('time').reset_index(drop=True)
        
        # Create time-based features
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['day_of_year'] = df['time'].dt.dayofyear
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['week_of_year'] = df['time'].dt.isocalendar().week
        
        # Season mapping
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        # Additional calculations with null handling
        df['temp_wetbulb_diff'] = df['Temp'] - df['WetBulb']
        
        # FIXED: More robust heat index calculation
        df['heat_index'] = np.where(
            (df['Temp'].notna()) & (df['Humidity'].notna()),
            0.5 * (df['Temp'] + 61.0 + ((df['Temp']-68.0)*1.2) + (df['Humidity']*0.094)),
            np.nan
        )
        
        # Rolling averages with minimum periods
        df['WetBulb_7day_avg'] = df['WetBulb'].rolling(window=24*7, center=True, min_periods=1).mean()
        df['WetBulb_30day_avg'] = df['WetBulb'].rolling(window=24*30, center=True, min_periods=1).mean()
        df['WetBulb_rolling_std'] = df['WetBulb'].rolling(window=24*7, center=True, min_periods=1).std()
        
        # Trend analysis with better error handling
        df['time_numeric'] = (df['time'] - df['time'].min()).dt.total_seconds()
        
        def safe_trend_calculation(x):
            try:
                if len(x) > 1 and x.notna().sum() > 1:
                    return stats.linregress(range(len(x)), x.fillna(x.mean()))[0]
                else:
                    return 0
            except:
                return 0
        
        df['WetBulb_trend'] = df['WetBulb'].rolling(window=24*30, center=True, min_periods=1).apply(safe_trend_calculation)
        
        # Add station name
        df['station'] = station_name
        
        return df
        
    except Exception as e:
        st.error(f"Error processing data for {station_name}: {str(e)}")
        return pd.DataFrame()

# Forecasting functions - FIXED: Added better error handling
def create_forecast(df, periods=168):  # 7 days (168 hours)
    """Create forecast using multiple methods"""
    try:
        df_clean = df.dropna(subset=['WetBulb', 'Temp', 'Humidity'])
        
        if len(df_clean) < 100:
            return None, "Insufficient data for forecasting (need at least 100 records)"
        
        # Prepare features
        feature_cols = ['hour', 'day_of_year', 'Temp', 'Humidity']
        X = df_clean[feature_cols].values
        y = df_clean['WetBulb'].values
        
        # Check for any remaining NaN values
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            return None, "Data contains NaN values after cleaning"
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Multiple models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                results[name] = {
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'predictions': y_pred
                }
            except Exception as e:
                st.warning(f"Model {name} failed: {str(e)}")
                continue
        
        if not results:
            return None, "All models failed to train"
        
        # Generate future predictions
        last_time = df_clean['time'].max()
        future_times = pd.date_range(start=last_time + timedelta(hours=1), periods=periods, freq='H')
        
        # Create future features (simplified)
        future_features = []
        for t in future_times:
            hour = t.hour
            day_of_year = t.dayofyear
            # Use recent averages for temp and humidity
            recent_temp = df_clean['Temp'].tail(168).mean()
            recent_humidity = df_clean['Humidity'].tail(168).mean()
            future_features.append([hour, day_of_year, recent_temp, recent_humidity])
        
        future_features = np.array(future_features)
        
        # Best model (lowest RMSE)
        best_model_name = min(results, key=lambda x: results[x]['rmse'])
        best_model = results[best_model_name]['model']
        
        future_predictions = best_model.predict(future_features)
        
        forecast_df = pd.DataFrame({
            'time': future_times,
            'WetBulb_forecast': future_predictions,
            'model': best_model_name
        })
        
        return forecast_df, results
        
    except Exception as e:
        return None, f"Forecasting error: {str(e)}"

# Chart creation functions - FIXED: Added better error handling and performance optimizations
def create_advanced_charts(df, chart_type, theme):
    """Create various advanced chart types"""
    
    try:
        if chart_type == "3D Surface Plot":
            # 3D surface plot of temperature vs humidity vs wet bulb
            # FIXED: Better sampling and null handling
            df_clean = df.dropna(subset=['Temp', 'Humidity', 'WetBulb'])
            if len(df_clean) == 0:
                raise ValueError("No valid data for 3D plot")
            
            sample_df = df_clean.sample(min(1000, len(df_clean)))  # Sample for performance
            
            fig = go.Figure(data=[go.Scatter3d(
                x=sample_df['Temp'],
                y=sample_df['Humidity'],
                z=sample_df['WetBulb'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=sample_df['WetBulb'],
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Wet Bulb (°C)")
                ),
                text=sample_df['time'].dt.strftime('%Y-%m-%d %H:%M'),
                hovertemplate='<b>Temp:</b> %{x:.1f}°C<br>' +
                             '<b>Humidity:</b> %{y:.1f}%<br>' +
                             '<b>Wet Bulb:</b> %{z:.1f}°C<br>' +
                             '<b>Time:</b> %{text}<extra></extra>'
            )])
            
            fig.update_layout(
                title='3D Relationship: Temperature vs Humidity vs Wet Bulb',
                scene=dict(
                    xaxis_title='Temperature (°C)',
                    yaxis_title='Humidity (%)',
                    zaxis_title='Wet Bulb (°C)'
                ),
                height=600,
                template='plotly_white'
            )
            
            return fig
        
        elif chart_type == "Heatmap Calendar":
            # FIXED: Better handling of calendar heatmap
            df_clean = df.dropna(subset=['WetBulb'])
            if len(df_clean) == 0:
                raise ValueError("No valid data for calendar heatmap")
            
            df_daily = df_clean.groupby(df_clean['time'].dt.date)['WetBulb'].mean().reset_index()
            df_daily['time'] = pd.to_datetime(df_daily['time'])
            df_daily['year'] = df_daily['time'].dt.year
            df_daily['month'] = df_daily['time'].dt.month
            df_daily['day'] = df_daily['time'].dt.day
            
            # Create pivot table for heatmap
            years = sorted(df_daily['year'].unique())
            
            fig = make_subplots(
                rows=len(years), cols=1,
                subplot_titles=[f'Year {year}' for year in years],
                vertical_spacing=0.1
            )
            
            for i, year in enumerate(years):
                year_data = df_daily[df_daily['year'] == year]
                
                # Create a matrix for the year
                year_matrix = np.full((12, 31), np.nan)
                for _, row in year_data.iterrows():
                    if 1 <= row['month'] <= 12 and 1 <= row['day'] <= 31:
                        year_matrix[row['month']-1, row['day']-1] = row['WetBulb']
                
                fig.add_trace(
                    go.Heatmap(
                        z=year_matrix,
                        colorscale='RdYlBu_r',
                        showscale=(i == 0),
                        colorbar=dict(title="Wet Bulb (°C)") if i == 0 else None,
                        name=f'Year {year}'
                    ),
                    row=i+1, col=1
                )
            
            fig.update_layout(
                title='Calendar Heatmap of Daily Average Wet Bulb Temperature',
                height=200 * len(years),
                template='plotly_white'
            )
            
            return fig
        
        elif chart_type == "Polar Chart":
            # FIXED: Better error handling for polar chart
            df_clean = df.dropna(subset=['WetBulb'])
            if len(df_clean) == 0:
                raise ValueError("No valid data for polar chart")
            
            monthly_avg = df_clean.groupby('month')['WetBulb'].mean()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=monthly_avg.values,
                theta=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                fill='toself',
                name='Wet Bulb Temperature',
                line_color=theme['primary_color']
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[monthly_avg.min()*0.9, monthly_avg.max()*1.1]
                    )
                ),
                title='Seasonal Pattern (Polar View)',
                template='plotly_white',
                height=500
            )
            
            return fig
        
        elif chart_type == "Violin Plot":
            # FIXED: Better handling of violin plot
            df_clean = df.dropna(subset=['WetBulb', 'season'])
            if len(df_clean) == 0:
                raise ValueError("No valid data for violin plot")
            
            fig = go.Figure()
            
            seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
            colors = [theme['primary_color'], theme['secondary_color'], 
                     theme['accent_color'], '#96CEB4']
            
            for season, color in zip(seasons, colors):
                season_data = df_clean[df_clean['season'] == season]['WetBulb']
                
                if len(season_data) > 0:
                    fig.add_trace(go.Violin(
                        y=season_data,
                        name=season,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=color,
                        opacity=0.7
                    ))
            
            fig.update_layout(
                title='Distribution of Wet Bulb Temperature by Season',
                yaxis_title='Wet Bulb Temperature (°C)',
                template='plotly_white',
                height=500
            )
            
            return fig
        
        elif chart_type == "Contour Plot":
            # FIXED: Better handling of contour plot
            df_clean = df.dropna(subset=['WetBulb', 'hour', 'day_of_year'])
            if len(df_clean) == 0:
                raise ValueError("No valid data for contour plot")
            
            pivot_data = df_clean.pivot_table(
                values='WetBulb', 
                index='hour', 
                columns='day_of_year', 
                aggfunc='mean'
            )
            
            if pivot_data.empty:
                raise ValueError("Cannot create pivot table for contour plot")
            
            fig = go.Figure(data=go.Contour(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='Viridis',
                colorbar=dict(title="Wet Bulb (°C)")
            ))
            
            fig.update_layout(
                title='Wet Bulb Temperature Contour: Hour vs Day of Year',
                xaxis_title='Day of Year',
                yaxis_title='Hour of Day',
                template='plotly_white',
                height=500
            )
            
            return fig
        
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")
            
    except Exception as e:
        # Return a simple error plot
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title=f"Error in {chart_type}",
            height=400,
            template='plotly_white'
        )
        return fig

# FIXED: Added main function structure for better organization
def main():
    # Sidebar theme selector
    st.sidebar.title("🎨 Dashboard Configuration")
    st.sidebar.markdown("---")
    
    selected_theme_name = st.sidebar.selectbox(
        "Choose Theme",
        options=list(THEMES.keys()),
        index=0
    )
    theme = THEMES[selected_theme_name]
    
    # Apply custom CSS
    apply_custom_css(theme)
    
    # Main title
    st.markdown('<h1 class="main-header">🌡️ Enhanced Wet Bulb Temperature Dashboard</h1>', unsafe_allow_html=True)
    
    # Station selector
    st.sidebar.markdown("### 📍 Station Selection")
    selected_stations = st.sidebar.multiselect(
        "Select Weather Stations",
        options=list(STATIONS.keys()),
        default=[list(STATIONS.keys())[0]],
        help="Choose one or more weather stations to analyze"
    )
    
    if not selected_stations:
        st.error("Please select at least one weather station.")
        return
    
    # Load data for selected stations
    combined_df = pd.DataFrame()
    loading_status = st.empty()
    
    for i, station in enumerate(selected_stations):
        loading_status.text(f"Loading data for {station}... ({i+1}/{len(selected_stations)})")
        
        df, error = load_station_data(STATIONS[station])
        
        if df is not None:
            processed_df = process_station_data(df, station)
            if not processed_df.empty:
                combined_df = pd.concat([combined_df, processed_df], ignore_index=True)
        else:
            st.error(f"Failed to load data for {station}: {error}")
    
    loading_status.empty()
    
    if combined_df.empty:
        st.error("No data could be loaded. Please check your station selection.")
        return
    
    st.success(f"✅ Successfully loaded data for {len(selected_stations)} station(s)! Total records: {len(combined_df):,}")
    
    # FIXED: Better date range handling
    if combined_df['time'].notna().any():
        min_date = combined_df['time'].min().date()
        max_date = combined_df['time'].max().date()
        
        # Sidebar controls
        st.sidebar.markdown("### 🎛️ Filter Controls")
        
        # Date range selector
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Ensure date_range is a tuple
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range
    else:
        st.error("No valid time data found")
        return
    
    # Station filter (if multiple selected)
    if len(selected_stations) > 1:
        station_filter = st.sidebar.multiselect(
            "Filter by Station",
            options=selected_stations,
            default=selected_stations
        )
    else:
        station_filter = selected_stations
    
    # Other filters
    available_years = sorted(combined_df['year'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Years",
        options=available_years,
        default=available_years[-3:] if len(available_years) > 3 else available_years
    )
    
    selected_seasons = st.sidebar.multiselect(
        "Select Seasons",
        options=['Spring', 'Summer', 'Autumn', 'Winter'],
        default=['Spring', 'Summer', 'Autumn', 'Winter']
    )
    
    hour_range = st.sidebar.slider(
        "Hour Range",
        min_value=0,
        max_value=23,
        value=(0, 23)
    )
    
    # Apply filters
    filtered_df = combined_df[
        (combined_df['time'].dt.date >= start_date) &
        (combined_df['time'].dt.date <= end_date) &
        (combined_df['station'].isin(station_filter)) &
        (combined_df['year'].isin(selected_years)) &
        (combined_df['season'].isin(selected_seasons)) &
        (combined_df['hour'] >= hour_range[0]) &
        (combined_df['hour'] <= hour_range[1])
    ].copy()
    
    if len(filtered_df) == 0:
        st.error("❌ No data available for the selected filters. Please adjust your selection.")
        return
    
    # Key metrics
    st.markdown("---")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("📊 Records", f"{len(filtered_df):,}")
    with col2:
        avg_wb = filtered_df['WetBulb'].mean()
        st.metric("🌡️ Avg Wet Bulb", f"{avg_wb:.1f}°C" if not np.isnan(avg_wb) else "N/A")
    with col3:
        max_wb = filtered_df['WetBulb'].max()
        st.metric("🔥 Max Wet Bulb", f"{max_wb:.1f}°C" if not np.isnan(max_wb) else "N/A")
    with col4:
        min_wb = filtered_df['WetBulb'].min()
        st.metric("🧊 Min Wet Bulb", f"{min_wb:.1f}°C" if not np.isnan(min_wb) else "N/A")
    with col5:
        std_wb = filtered_df['WetBulb'].std()
        st.metric("📈 Std Dev", f"{std_wb:.1f}°C" if not np.isnan(std_wb) else "N/A")
    with col6:
        st.metric("📍 Stations", len(station_filter))

if __name__ == "__main__":
    main()