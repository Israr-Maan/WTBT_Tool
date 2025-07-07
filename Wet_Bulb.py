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

# Station configuration - Update these URLs to your actual Git repository
GITHUB_BASE_URL = "https://raw.githubusercontent.com/your-username/your-repo/main/data/"
STATIONS = {
    "Station_01_Karachi": "station_01_karachi.csv",
    "Station_02_Lahore": "station_02_lahore.csv",
    "Station_03_Islamabad": "station_03_islamabad.csv",
    "Station_04_Peshawar": "station_04_peshawar.csv",
    "Station_05_Quetta": "station_05_quetta.csv",
    "Station_06_Multan": "station_06_multan.csv",
    "Station_07_Faisalabad": "station_07_faisalabad.csv",
    "Station_08_Hyderabad": "station_08_hyderabad.csv",
    "Station_09_Gujranwala": "station_09_gujranwala.csv",
    "Station_10_Sialkot": "station_10_sialkot.csv",
    "Station_11_Sargodha": "station_11_sargodha.csv",
    "Station_12_Bahawalpur": "station_12_bahawalpur.csv",
    "Station_13_Sukkur": "station_13_sukkur.csv",
    "Station_14_Larkana": "station_14_larkana.csv",
    "Station_15_Nawabshah": "station_15_nawabshah.csv",
    "Station_16_Jacobabad": "station_16_jacobabad.csv"
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

# Load data from GitHub
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_station_data(station_file):
    """Load data from GitHub repository"""
    try:
        url = GITHUB_BASE_URL + station_file
        response = requests.get(url)
        response.raise_for_status()
        
        # Read CSV from string content
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        return df, None
    except Exception as e:
        return None, str(e)

# Enhanced data processing
@st.cache_data
def process_station_data(df, station_name):
    """Enhanced data preprocessing with additional features"""
    df = df.copy()
    
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])
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
    
    # Additional calculations
    df['temp_wetbulb_diff'] = df['Temp'] - df['WetBulb']
    df['heat_index'] = 0.5 * (df['Temp'] + 61.0 + ((df['Temp']-68.0)*1.2) + (df['Humidity']*0.094))
    
    # Rolling averages
    df['WetBulb_7day_avg'] = df['WetBulb'].rolling(window=24*7, center=True).mean()
    df['WetBulb_30day_avg'] = df['WetBulb'].rolling(window=24*30, center=True).mean()
    df['WetBulb_rolling_std'] = df['WetBulb'].rolling(window=24*7, center=True).std()
    
    # Trend analysis
    df['time_numeric'] = (df['time'] - df['time'].min()).dt.total_seconds()
    df['WetBulb_trend'] = df['WetBulb'].rolling(window=24*30, center=True).apply(
        lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0
    )
    
    # Add station name
    df['station'] = station_name
    
    return df

# Forecasting functions
def create_forecast(df, periods=168):  # 7 days (168 hours)
    """Create forecast using multiple methods"""
    df_clean = df.dropna(subset=['WetBulb'])
    
    if len(df_clean) < 100:
        return None, "Insufficient data for forecasting"
    
    # Prepare features
    X = df_clean[['hour', 'day_of_year', 'Temp', 'Humidity']].values
    y = df_clean['WetBulb'].values
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
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

# Chart creation functions
def create_advanced_charts(df, chart_type, theme):
    """Create various advanced chart types"""
    
    if chart_type == "3D Surface Plot":
        # 3D surface plot of temperature vs humidity vs wet bulb
        sample_df = df.sample(min(1000, len(df)))  # Sample for performance
        
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
        # Calendar heatmap
        df_daily = df.groupby(df['time'].dt.date)['WetBulb'].mean().reset_index()
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
                year_matrix[row['month']-1, row['day']-1] = row['WetBulb']
            
            fig.add_trace(
                go.Heatmap(
                    z=year_matrix,
                    colorscale='RdYlBu_r',
                    showscale=(i == 0),
                    colorbar=dict(title="Wet Bulb (°C)") if i == 0 else None
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
        # Polar chart showing seasonal patterns
        monthly_avg = df.groupby('month')['WetBulb'].mean()
        
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
        # Violin plot by season
        fig = go.Figure()
        
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        colors = [theme['primary_color'], theme['secondary_color'], 
                 theme['accent_color'], '#96CEB4']
        
        for season, color in zip(seasons, colors):
            season_data = df[df['season'] == season]['WetBulb']
            
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
        # Contour plot of hour vs day of year
        pivot_data = df.pivot_table(
            values='WetBulb', 
            index='hour', 
            columns='day_of_year', 
            aggfunc='mean'
        )
        
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

# Main application
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
            combined_df = pd.concat([combined_df, processed_df], ignore_index=True)
        else:
            st.error(f"Failed to load data for {station}: {error}")
    
    loading_status.empty()
    
    if combined_df.empty:
        st.error("No data could be loaded. Please check your station selection.")
        return
    
    st.success(f"✅ Successfully loaded data for {len(selected_stations)} station(s)! Total records: {len(combined_df):,}")
    
    # Sidebar controls
    st.sidebar.markdown("### 🎛️ Filter Controls")
    
    # Date range selector
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(combined_df['time'].min().date(), combined_df['time'].max().date()),
        min_value=combined_df['time'].min().date(),
        max_value=combined_df['time'].max().date()
    )
    
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
        (combined_df['time'].dt.date >= date_range[0]) &
        (combined_df['time'].dt.date <= date_range[1]) &
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
        st.metric("🌡️ Avg Wet Bulb", f"{filtered_df['WetBulb'].mean():.1f}°C")
    with col3:
        st.metric("🔥 Max Wet Bulb", f"{filtered_df['WetBulb'].max():.1f}°C")
    with col4:
        st.metric("🧊 Min Wet Bulb", f"{filtered_df['WetBulb'].min():.1f}°C")
    with col5:
        st.metric("📈 Std Dev", f"{filtered_df['WetBulb'].std():.1f}°C")
    with col6:
        st.metric("📍 Stations", len(station_filter))
    
    # Enhanced tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Time Series", "🎨 Advanced Charts", "🔮 Forecasting", 
        "🌤️ Seasonal Analysis", "🔥 Extreme Events", "📊 Statistics", "📋 Data Export"
    ])
    
    with tab1:
        st.markdown("### 📈 Interactive Time Series Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            chart_options = st.expander("Chart Options", expanded=True)
            with chart_options:
                show_temp = st.checkbox("Show Temperature", value=True)
                show_humidity = st.checkbox("Show Humidity", value=True)
                show_moving_avg = st.checkbox("Show Moving Averages", value=False)
                show_trend = st.checkbox("Show Trend Lines", value=False)
                aggregation = st.selectbox(
                    "Data Aggregation",
                    options=["Hourly", "Daily", "Weekly", "Monthly"],
                    index=1
                )
                
                if len(selected_stations) > 1:
                    compare_stations = st.checkbox("Compare Stations", value=True)
                else:
                    compare_stations = False
        
        with col1:
            # Create enhanced time series plot
            fig_ts = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Wet Bulb Temperature', 'Temperature & Humidity'),
                vertical_spacing=0.12,
                specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
            )
            
            # Data aggregation
            if aggregation == "Daily":
                if compare_stations:
                    plot_df = filtered_df.groupby(['station', filtered_df['time'].dt.date]).mean().reset_index()
                else:
                    plot_df = filtered_df.groupby(filtered_df['time'].dt.date).mean().reset_index()
                plot_df['time'] = pd.to_datetime(plot_df['time'])
            elif aggregation == "Weekly":
                if compare_stations:
                    plot_df = filtered_df.groupby(['station', pd.Grouper(key='time', freq='W')]).mean().reset_index()
                else:
                    plot_df = filtered_df.groupby(pd.Grouper(key='time', freq='W')).mean().reset_index()
            elif aggregation == "Monthly":
                if compare_stations:
                    plot_df = filtered_df.groupby(['station', pd.Grouper(key='time', freq='M')]).mean().reset_index()
                else:
                    plot_df = filtered_df.groupby(pd.Grouper(key='time', freq='M')).mean().reset_index()
            else:
                plot_df = filtered_df.copy()
            
            # Plot wet bulb temperature
            if compare_stations and len(selected_stations) > 1:
                colors = [theme['primary_color'], theme['secondary_color'], theme['accent_color']] * 10
                for i, station in enumerate(station_filter):
                    station_data = plot_df[plot_df['station'] == station]
                    fig_ts.add_trace(
                        go.Scatter(
                            x=station_data['time'], 
                            y=station_data['WetBulb'],
                            mode='lines', 
                            name=f'{station} - Wet Bulb',
                            line=dict(color=colors[i], width=2)
                        ),
                        row=1, col=1
                    )
            else:
                fig_ts.add_trace(
                    go.Scatter(
                        x=plot_df['time'], 
                        y=plot_df['WetBulb'],
                        mode='lines', 
                        name='Wet Bulb Temperature',
                        line=dict(color=theme['primary_color'], width=2)
                    ),
                    row=1, col=1
                )
            
            # Add moving averages and trends if requested
            if show_moving_avg and 'WetBulb_7day_avg' in plot_df.columns:
                fig_ts.add_trace(
                    go.Scatter(
                        x=plot_df['time'], 
                        y=plot_df['WetBulb_7day_avg'],
                        mode='lines', 
                        name='7-day Average',
                        line=dict(color=theme['primary_color'], width=1, dash='dash')
                    ),
                    row=1, col=1
                )
            
            if show_trend:
                # Add trend line
                x_numeric = (plot_df['time'] - plot_df['time'].min()).dt.total_seconds()
                slope, intercept, _, _, _ = stats.linregress(x_numeric, plot_df['WetBulb'])
                trend_line = slope * x_numeric + intercept
                
                fig_ts.add_trace(
                    go.Scatter(
                        x=plot_df['time'], 
                        y=trend_line,
                        mode='lines', 
                        name='Trend Line',
                        line=dict(color='red', width=2, dash='dot')
                    ),
                    row=1, col=1
                )
            
            # Temperature and humidity plots
            if show_temp:
                fig_ts.add_trace(
                    go.Scatter(
                        x=plot_df['time'], 
                        y=plot_df['Temp'],
                        mode='lines', 
                        name='Temperature',
                        line=dict(color=theme['secondary_color'], width=1.5)
                    ),
                    row=2, col=1
                )
            
            if show_humidity:
                fig_ts.add_trace(
                    go.Scatter(
                        x=plot_df['time'], 
                        y=plot_df['Humidity'],
                        mode='lines', 
                        name='Humidity',
                        line=dict(color=theme['accent_color'], width=1.5),
                        yaxis='y3'
                    ),
                    row=2, col=1
                )
            
            fig_ts.update_layout(
                height=700, 
                template='plotly_white',
                hovermode='x unified'
            )
            fig_ts.update_yaxes(title_text="Wet Bulb Temperature (°C)", row=1, col=1)
            fig_ts.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
            fig_ts.update_yaxes(title_text="Humidity (%)", secondary_y=True, row=2, col=1)
            
            st.plotly_chart(fig_ts, use_container_width=True)
    
    with tab2:
        st.markdown("### 🎨 Advanced Chart Types")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            chart_type = st.selectbox(
                "Select Chart Type",
                options=[
                    "3D Surface Plot",
                    "Heatmap Calendar", 
                    "Polar Chart",
                    "Violin Plot",
                    "Contour Plot"
                ]
            )
            
            st.markdown("**Chart Description:**")
            descriptions = {
                "3D Surface Plot": "3D visualization of temperature, humidity, and wet bulb relationships",
                "Heatmap Calendar": "Calendar view showing daily average wet bulb temperatures",
                "Polar Chart": "Circular visualization of seasonal patterns",
                "Violin Plot": "Distribution comparison across seasons",
                "Contour Plot": "2D density map of hourly and seasonal patterns"
            }
            
            st.info(descriptions[chart_type])
        
        with col2:
            with st.spinner(f"Creating {chart_type}..."):
                try:
                    advanced_fig = create_advanced_charts(filtered_df, chart_type, theme)
                    st.plotly_chart(advanced_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating chart: {str(e)}")
    
    with tab3:
        st.markdown("### 🔮 Forecasting & Predictions")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Forecast Settings")
            forecast_days = st.slider("Forecast Period (days)", 1, 30, 7)
            forecast_station = st.selectbox(
                "Select Station for Forecast",
                options=station_filter,
                help="Choose a single station for detailed forecasting"
            )
            
            confidence_interval = st.checkbox("Show Confidence Intervals", value=True)
            
            if st.button("🔮 Generate Forecast", type="primary"):
                station_data = filtered_df[filtered_df['station'] == forecast_station].copy()
                
                with st.spinner("Generating forecast..."):
                    forecast_df, model_results = create_forecast(station_data, periods=forecast_days*24)
                
                if forecast_df is not None:
                    st.session_state['forecast_data'] = forecast_df
                    st.session_state['model_results'] = model_results
                    st.success("✅ Forecast generated successfully!")
                else:
                    st.error("❌ Unable to generate forecast. Insufficient data.")
        
        with col2:
            if 'forecast_data' in st.session_state:
                forecast_df = st.session_state['forecast_data']
                model_results = st.session_state['model_results']
                
                # Create forecast plot
                fig_forecast = go.Figure()
                
                # Historical data (last 30 days)
                station_data = filtered_df[filtered_df['station'] == forecast_station].copy()
                recent_data = station_data.tail(24*30)  # Last 30 days
                
                fig_forecast.add_trace(
                    go.Scatter(
                        x=recent_data['time'],
                        y=recent_data['WetBulb'],
                        mode='lines',
                        name='Historical Data',
                        line=dict(color=theme['primary_color'], width=2)
                    )
                )
                
                # Forecast
                fig_forecast.add_trace(
                    go.Scatter(
                        x=forecast_df['time'],
                        y=forecast_df['WetBulb_forecast'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color=theme['secondary_color'], width=2, dash='dash')
                    )
                )
                
                # Confidence intervals (simplified)
                if confidence_interval:
                    forecast_std = station_data['WetBulb'].std()
                    upper_bound = forecast_df['WetBulb_forecast'] + 1.96 * forecast_std
                    lower_bound = forecast_df['WetBulb_forecast'] - 1.96 * forecast_std
                    
                    fig_forecast.add_trace(
                        go.Scatter(
                            x=forecast_df['time'],
                            y=upper_bound,
                            fill=None,
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            showlegend=False
                        )
                    )
                    
                    fig_forecast.add_trace(
                        go.Scatter(
                            x=forecast_df['time'],
                            y=lower_bound,
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            name='95% Confidence',
                            fillcolor=f'rgba({theme["secondary_color"][1:]}, 0.3)'
                        )
                    )
                
                # Add vertical line to separate historical and forecast
                fig_forecast.add_vline(
                    x=recent_data['time'].max(),
                    line_dash="dot",
                    line_color="gray",
                    annotation_text="Forecast Start"
                )
                
                fig_forecast.update_layout(
                    title=f'Wet Bulb Temperature Forecast - {forecast_station}',
                    xaxis_title='Time',
                    yaxis_title='Wet Bulb Temperature (°C)',
                    height=500,
                    template='plotly_white',
                    hovermode='x'
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Model performance metrics
                st.markdown("#### 📊 Model Performance")
                perf_cols = st.columns(len(model_results))
                
                for i, (model_name, results) in enumerate(model_results.items()):
                    with perf_cols[i]:
                        st.metric(
                            f"{model_name} MAE",
                            f"{results['mae']:.2f}°C"
                        )
                        st.metric(
                            f"{model_name} RMSE",
                            f"{results['rmse']:.2f}°C"
                        )
            else:
                st.info("👆 Configure settings and click 'Generate Forecast' to see predictions")
    
    with tab4:
        st.markdown("### 🌤️ Enhanced Seasonal Analysis")
        
        # Seasonal statistics
        seasonal_stats = filtered_df.groupby(['season', 'station'])['WetBulb'].agg(['mean', 'std', 'min', 'max']).round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced seasonal boxplot with violin overlay
            fig_seasonal = go.Figure()
            
            seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
            colors = [theme['primary_color'], theme['secondary_color'], theme['accent_color'], '#96CEB4']
            
            for season, color in zip(seasons, colors):
                season_data = filtered_df[filtered_df['season'] == season]['WetBulb']
                
                # Add violin plot
                fig_seasonal.add_trace(go.Violin(
                    y=season_data,
                    name=season,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=color,
                    opacity=0.7,
                    side='positive',
                    width=0.8
                ))
            
            fig_seasonal.update_layout(
                title='Seasonal Distribution with Statistical Details',
                yaxis_title='Wet Bulb Temperature (°C)',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_seasonal, use_container_width=True)
            
            # Seasonal trends over years
            yearly_seasonal = filtered_df.groupby(['year', 'season'])['WetBulb'].mean().unstack(fill_value=0)
            
            fig_trend = go.Figure()
            
            for season, color in zip(seasons, colors):
                if season in yearly_seasonal.columns:
                    fig_trend.add_trace(go.Scatter(
                        x=yearly_seasonal.index,
                        y=yearly_seasonal[season],
                        mode='lines+markers',
                        name=season,
                        line=dict(color=color, width=3)
                    ))
            
            fig_trend.update_layout(
                title='Seasonal Trends Over Years',
                xaxis_title='Year',
                yaxis_title='Average Wet Bulb Temperature (°C)',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # Monthly patterns with error bars
            monthly_stats = filtered_df.groupby('month')['WetBulb'].agg(['mean', 'std']).reset_index()
            monthly_stats['month_name'] = monthly_stats['month'].map({
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            })
            
            fig_monthly = go.Figure()
            
            fig_monthly.add_trace(go.Scatter(
                x=monthly_stats['month_name'],
                y=monthly_stats['mean'],
                error_y=dict(
                    type='data',
                    array=monthly_stats['std'],
                    visible=True
                ),
                mode='lines+markers',
                name='Monthly Average',
                line=dict(color=theme['primary_color'], width=3),
                marker=dict(size=8)
            ))
            
            fig_monthly.update_layout(
                title='Monthly Patterns with Standard Deviation',
                xaxis_title='Month',
                yaxis_title='Wet Bulb Temperature (°C)',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Diurnal patterns by season
            diurnal_seasonal = filtered_df.groupby(['hour', 'season'])['WetBulb'].mean().unstack()
            
            fig_diurnal = go.Figure()
            
            for season, color in zip(seasons, colors):
                if season in diurnal_seasonal.columns:
                    fig_diurnal.add_trace(go.Scatter(
                        x=diurnal_seasonal.index,
                        y=diurnal_seasonal[season],
                        mode='lines+markers',
                        name=season,
                        line=dict(color=color, width=2)
                    ))
            
            fig_diurnal.update_layout(
                title='Diurnal Patterns by Season',
                xaxis_title='Hour of Day',
                yaxis_title='Average Wet Bulb Temperature (°C)',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_diurnal, use_container_width=True)
        
        # Seasonal statistics table
        st.markdown("#### 📊 Detailed Seasonal Statistics")
        st.dataframe(seasonal_stats, use_container_width=True)
    
    with tab5:
        st.markdown("### 🔥 Advanced Extreme Events Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Extreme Event Settings")
            
            extreme_method = st.selectbox(
                "Detection Method",
                options=["Percentile", "Standard Deviation", "Absolute Threshold"],
                help="Choose method for defining extreme events"
            )
            
            if extreme_method == "Percentile":
                extreme_threshold = st.slider("Percentile Threshold", 90, 99, 95)
                wb_high = filtered_df['WetBulb'].quantile(extreme_threshold/100)
                wb_low = filtered_df['WetBulb'].quantile((100-extreme_threshold)/100)
            elif extreme_method == "Standard Deviation":
                std_multiplier = st.slider("Standard Deviations", 1.5, 3.0, 2.0, 0.1)
                mean_wb = filtered_df['WetBulb'].mean()
                std_wb = filtered_df['WetBulb'].std()
                wb_high = mean_wb + std_multiplier * std_wb
                wb_low = mean_wb - std_multiplier * std_wb
            else:  # Absolute Threshold
                wb_high = st.number_input("High Threshold (°C)", value=30.0, step=0.1)
                wb_low = st.number_input("Low Threshold (°C)", value=10.0, step=0.1)
            
            min_duration = st.slider("Minimum Event Duration (hours)", 1, 24, 3)
            
            st.info(f"**High Threshold:** {wb_high:.1f}°C  \n**Low Threshold:** {wb_low:.1f}°C")
        
        with col2:
            # Identify extreme events
            extreme_high = filtered_df[filtered_df['WetBulb'] >= wb_high].copy()
            extreme_low = filtered_df[filtered_df['WetBulb'] <= wb_low].copy()
            
            # Group consecutive events
            def group_consecutive_events(df, min_duration_hours):
                if len(df) == 0:
                    return pd.DataFrame()
                
                df = df.sort_values('time').reset_index(drop=True)
                df['time_diff'] = df['time'].diff().dt.total_seconds() / 3600  # hours
                df['event_group'] = (df['time_diff'] > 2).cumsum()  # New event if gap > 2 hours
                
                # Filter events by minimum duration
                event_summary = df.groupby('event_group').agg({
                    'time': ['min', 'max', 'count'],
                    'WetBulb': ['mean', 'max', 'min'],
                    'station': 'first'
                }).round(2)
                
                event_summary.columns = ['start_time', 'end_time', 'duration_hours', 
                                       'avg_wetbulb', 'max_wetbulb', 'min_wetbulb', 'station']
                
                # Filter by minimum duration
                event_summary = event_summary[event_summary['duration_hours'] >= min_duration_hours]
                
                return event_summary
            
            high_events = group_consecutive_events(extreme_high, min_duration)
            low_events = group_consecutive_events(extreme_low, min_duration)
            
            # Metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("🔥 High Events", len(high_events))
            with metrics_col2:
                st.metric("🧊 Low Events", len(low_events))
            with metrics_col3:
                total_extreme_hours = len(extreme_high) + len(extreme_low)
                extreme_percentage = (total_extreme_hours / len(filtered_df)) * 100
                st.metric("📊 Extreme %", f"{extreme_percentage:.1f}%")
            
            # Extreme events timeline
            fig_extreme = go.Figure()
            
            # Main time series (sampled for performance)
            if len(filtered_df) > 10000:
                sample_df = filtered_df.sample(10000).sort_values('time')
            else:
                sample_df = filtered_df
            
            fig_extreme.add_trace(
                go.Scatter(
                    x=sample_df['time'],
                    y=sample_df['WetBulb'],
                    mode='lines',
                    name='Wet Bulb Temperature',
                    line=dict(color='lightgray', width=1),
                    opacity=0.7
                )
            )
            
            # Extreme events
            if len(extreme_high) > 0:
                fig_extreme.add_trace(
                    go.Scatter(
                        x=extreme_high['time'],
                        y=extreme_high['WetBulb'],
                        mode='markers',
                        name=f'Extreme High (≥{wb_high:.1f}°C)',
                        marker=dict(color='red', size=6, symbol='triangle-up')
                    )
                )
            
            if len(extreme_low) > 0:
                fig_extreme.add_trace(
                    go.Scatter(
                        x=extreme_low['time'],
                        y=extreme_low['WetBulb'],
                        mode='markers',
                        name=f'Extreme Low (≤{wb_low:.1f}°C)',
                        marker=dict(color='blue', size=6, symbol='triangle-down')
                    )
                )
            
            # Threshold lines
            fig_extreme.add_hline(y=wb_high, line_dash="dash", line_color="red", opacity=0.7)
            fig_extreme.add_hline(y=wb_low, line_dash="dash", line_color="blue", opacity=0.7)
            
            fig_extreme.update_layout(
                title='Extreme Events Timeline',
                xaxis_title='Time',
                yaxis_title='Wet Bulb Temperature (°C)',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_extreme, use_container_width=True)
        
        # Event details tables
        if len(high_events) > 0 or len(low_events) > 0:
            st.markdown("#### 📋 Extreme Event Details")
            
            event_col1, event_col2 = st.columns(2)
            
            with event_col1:
                if len(high_events) > 0:
                    st.markdown("**🔥 High Temperature Events**")
                    st.dataframe(high_events.head(10), use_container_width=True)
            
            with event_col2:
                if len(low_events) > 0:
                    st.markdown("**🧊 Low Temperature Events**")
                    st.dataframe(low_events.head(10), use_container_width=True)
    
    with tab6:
        st.markdown("### 📊 Advanced Statistical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced correlation analysis
            corr_vars = ['Temp', 'Humidity', 'WetBulb', 'heat_index', 'temp_wetbulb_diff']
            available_vars = [var for var in corr_vars if var in filtered_df.columns]
            
            corr_matrix = filtered_df[available_vars].corr()
            
            # Create interactive correlation heatmap
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(3),
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False
            ))
            
            fig_corr.update_layout(
                title='Enhanced Correlation Matrix',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Distribution analysis
            st.markdown("#### 📈 Distribution Analysis")
            
            dist_var = st.selectbox(
                "Select Variable for Distribution Analysis",
                options=available_vars,
                index=available_vars.index('WetBulb') if 'WetBulb' in available_vars else 0
            )
            
            # Create distribution plot with multiple visualizations
            fig_dist = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Histogram', 'Box Plot', 'Q-Q Plot', 'Density'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            data_var = filtered_df[dist_var].dropna()
            
            # Histogram
            fig_dist.add_trace(
                go.Histogram(x=data_var, nbinsx=50, name='Histogram'),
                row=1, col=1
            )
            
            # Box plot
            fig_dist.add_trace(
                go.Box(y=data_var, name='Box Plot'),
                row=1, col=2
            )
            
            # Q-Q plot
            from scipy.stats import probplot
            qq_x, qq_y = probplot(data_var, dist="norm", plot=None)
            fig_dist.add_trace(
                go.Scatter(x=qq_x, y=qq_y, mode='markers', name='Q-Q Plot'),
                row=2, col=1
            )
            
            # Add diagonal line for Q-Q plot
            min_val, max_val = min(qq_x), max(qq_x)
            fig_dist.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                          mode='lines', name='Normal Line', line=dict(color='red')),
                row=2, col=1
            )
            
            # Density plot
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data_var)
            x_range = np.linspace(data_var.min(), data_var.max(), 100)
            density = kde(x_range)
            
            fig_dist.add_trace(
                go.Scatter(x=x_range, y=density, mode='lines', name='Density'),
                row=2, col=2
            )
            
            fig_dist.update_layout(height=600, template='plotly_white')
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Statistical tests and metrics
            st.markdown("#### 🧮 Statistical Tests & Metrics")
            
            # Normality tests
            from scipy.stats import shapiro, normaltest
            
            wb_data = filtered_df['WetBulb'].dropna()
            
            if len(wb_data) > 5000:
                # Sample for large datasets
                wb_sample = wb_data.sample(5000)
            else:
                wb_sample = wb_data
            
            # Shapiro-Wilk test (for smaller samples)
            if len(wb_sample) <= 5000:
                shapiro_stat, shapiro_p = shapiro(wb_sample)
                st.metric("Shapiro-Wilk Test", f"p = {shapiro_p:.4f}")
                if shapiro_p > 0.05:
                    st.success("✅ Data appears normally distributed")
                else:
                    st.warning("⚠️ Data deviates from normal distribution")
            
            # D'Agostino's normality test
            dagostino_stat, dagostino_p = normaltest(wb_sample)
            st.metric("D'Agostino Test", f"p = {dagostino_p:.4f}")
            
            # Descriptive statistics
            st.markdown("#### 📊 Descriptive Statistics")
            
            stats_df = filtered_df[available_vars].describe().round(3)
            
            # Add additional statistics
            additional_stats = pd.DataFrame({
                var: {
                    'skewness': filtered_df[var].skew(),
                    'kurtosis': filtered_df[var].kurtosis(),
                    'range': filtered_df[var].max() - filtered_df[var].min(),
                    'iqr': filtered_df[var].quantile(0.75) - filtered_df[var].quantile(0.25)
                } for var in available_vars
            }).round(3)
            
            combined_stats = pd.concat([stats_df, additional_stats])
            st.dataframe(combined_stats, use_container_width=True)
            
            # Time series analysis
            if len(filtered_df) > 100:
                st.markdown("#### 📈 Time Series Properties")
                
                # Autocorrelation (simplified)
                wb_series = filtered_df.set_index('time')['WetBulb'].resample('D').mean().dropna()
                
                if len(wb_series) > 30:
                    # Calculate basic autocorrelation
                    autocorr_1 = wb_series.autocorr(lag=1)
                    autocorr_7 = wb_series.autocorr(lag=7) if len(wb_series) > 7 else None
                    autocorr_30 = wb_series.autocorr(lag=30) if len(wb_series) > 30 else None
                    
                    st.metric("1-day Autocorrelation", f"{autocorr_1:.3f}")
                    if autocorr_7 is not None:
                        st.metric("7-day Autocorrelation", f"{autocorr_7:.3f}")
                    if autocorr_30 is not None:
                        st.metric("30-day Autocorrelation", f"{autocorr_30:.3f}")
                    
                    # Trend detection
                    if 'WetBulb_trend' in filtered_df.columns:
                        avg_trend = filtered_df['WetBulb_trend'].mean()
                        st.metric("Average Trend (°C/month)", f"{avg_trend*24*30:.4f}")
    
    with tab7:
        st.markdown("### 📋 Data Export & Summary Report")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📊 Analysis Summary Report")
            
            # Generate comprehensive summary
            summary_report = f"""
# Wet Bulb Temperature Analysis Report

## Dataset Overview
- **Stations Analyzed:** {', '.join(station_filter)}
- **Analysis Period:** {filtered_df['time'].min().strftime('%Y-%m-%d')} to {filtered_df['time'].max().strftime('%Y-%m-%d')}
- **Total Records:** {len(filtered_df):,}
- **Data Quality:** {((len(filtered_df) - filtered_df['WetBulb'].isna().sum()) / len(filtered_df) * 100):.1f}% complete

## Key Statistics
- **Mean Wet Bulb Temperature:** {filtered_df['WetBulb'].mean():.2f}°C
- **Standard Deviation:** {filtered_df['WetBulb'].std():.2f}°C
- **Temperature Range:** {filtered_df['WetBulb'].min():.1f}°C to {filtered_df['WetBulb'].max():.1f}°C
- **Coefficient of Variation:** {(filtered_df['WetBulb'].std() / filtered_df['WetBulb'].mean() * 100):.1f}%

## Seasonal Analysis
"""
            
            # Add seasonal statistics to report
            seasonal_summary = filtered_df.groupby('season')['WetBulb'].agg(['mean', 'std']).round(2)
            for season in seasonal_summary.index:
                summary_report += f"- **{season}:** {seasonal_summary.loc[season, 'mean']:.1f}°C (σ = {seasonal_summary.loc[season, 'std']:.1f}°C)\n"
            
            # Add extreme events summary
            wb_95 = filtered_df['WetBulb'].quantile(0.95)
            wb_5 = filtered_df['WetBulb'].quantile(0.05)
            extreme_high_count = len(filtered_df[filtered_df['WetBulb'] >= wb_95])
            extreme_low_count = len(filtered_df[filtered_df['WetBulb'] <= wb_5])
            
            summary_report += f"""
## Extreme Events (95th/5th Percentile)
- **High Threshold:** {wb_95:.1f}°C ({extreme_high_count} events, {extreme_high_count/len(filtered_df)*100:.1f}%)
- **Low Threshold:** {wb_5:.1f}°C ({extreme_low_count} events, {extreme_low_count/len(filtered_df)*100:.1f}%)

## Correlations
- **Temperature vs Wet Bulb:** {filtered_df['Temp'].corr(filtered_df['WetBulb']):.3f}
- **Humidity vs Wet Bulb:** {filtered_df['Humidity'].corr(filtered_df['WetBulb']):.3f}

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            st.markdown(summary_report)
        
        with col2:
            st.markdown("#### 💾 Export Options")
            
            # Export format selection
            export_format = st.selectbox(
                "Export Format",
                options=["CSV", "Excel", "JSON", "Parquet"]
            )
            
            # Data selection options
            include_calculated = st.checkbox("Include Calculated Fields", value=True)
            include_forecast = st.checkbox("Include Forecast Data", value=False, 
                                         disabled='forecast_data' not in st.session_state)
            
            # Aggregation options
            export_aggregation = st.selectbox(
                "Data Aggregation",
                options=["Raw Data", "Hourly", "Daily", "Weekly", "Monthly"]
            )
            
            if st.button("📥 Download Data", type="primary"):
                # Prepare export data
                export_df = filtered_df.copy()
                
                # Apply aggregation
                if export_aggregation != "Raw Data":
                    freq_map = {
                        "Hourly": "H",
                        "Daily": "D", 
                        "Weekly": "W",
                        "Monthly": "M"
                    }
                    freq = freq_map[export_aggregation]
                    
                    export_df = export_df.groupby([pd.Grouper(key='time', freq=freq), 'station']).agg({
                        'Temp': 'mean',
                        'Humidity': 'mean', 
                        'WetBulb': 'mean',
                        'heat_index': 'mean' if 'heat_index' in export_df.columns else lambda x: x.iloc[0],
                        'temp_wetbulb_diff': 'mean' if 'temp_wetbulb_diff' in export_df.columns else lambda x: x.iloc[0]
                    }).reset_index()
                
                # Include forecast data if requested
                if include_forecast and 'forecast_data' in st.session_state:
                    forecast_df = st.session_state['forecast_data'].copy()
                    forecast_df['station'] = forecast_station
            if st.button("📥 Download Data", type="primary"):
                # Prepare export data
                export_df = filtered_df.copy()

                # Apply aggregation
                if export_aggregation != "Raw Data":
                    freq_map = {
                        "Hourly": "H",
                        "Daily": "D", 
                        "Weekly": "W",
                        "Monthly": "M"
                    }
                    freq = freq_map[export_aggregation]

                    export_df = export_df.groupby([pd.Grouper(key='time', freq=freq), 'station']).agg({
                        'Temp': 'mean',
                        'Humidity': 'mean', 
                        'WetBulb': 'mean',
                        'heat_index': 'mean' if 'heat_index' in export_df.columns else 'first',
                        'temp_wetbulb_diff': 'mean' if 'temp_wetbulb_diff' in export_df.columns else 'first'
                    }).reset_index()

                # Include forecast data if requested
                if include_forecast and 'forecast_data' in st.session_state:
                    forecast_df = st.session_state['forecast_data'].copy()
                    forecast_df['station'] = forecast_station  # Ensure station column exists
                    forecast_df = forecast_df.rename(columns={'WetBulb_forecast': 'WetBulb'})
                    export_df = pd.concat([export_df, forecast_df], ignore_index=True)

                # Select columns to export
                export_columns = ['time', 'station', 'Temp', 'Humidity', 'WetBulb']
                if include_calculated:
                    if 'heat_index' in export_df.columns:
                        export_columns.append('heat_index')
                    if 'temp_wetbulb_diff' in export_df.columns:
                        export_columns.append('temp_wetbulb_diff')

                export_df = export_df[export_columns]

                # Convert and export
                if export_format == "CSV":
                    csv = export_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", csv, "wetbulb_data.csv", mime="text/csv")

                elif export_format == "Excel":
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        export_df.to_excel(writer, index=False, sheet_name="WetBulb Data")
                        writer.save()
                    st.download_button("Download Excel", output.getvalue(), "wetbulb_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                elif export_format == "JSON":
                    json_data = export_df.to_json(orient="records", date_format='iso')
                    st.download_button("Download JSON", json_data, "wetbulb_data.json", mime="application/json")

                elif export_format == "Parquet":
                    from io import BytesIO
                    buffer = BytesIO()
                    export_df.to_parquet(buffer, index=False)
                    st.download_button("Download Parquet", buffer.getvalue(), "wetbulb_data.parquet", mime="application/octet-stream")
