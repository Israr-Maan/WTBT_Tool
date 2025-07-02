# Interactive Wet Bulb Temperature Dashboard
# Streamlit Application

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
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Wet Bulb Temperature Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2C3E50;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🌡️ Wet Bulb Temperature Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for file upload and controls
st.sidebar.title("📊 Dashboard Controls")
st.sidebar.markdown("---")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=['csv'],
    help="Upload a CSV file with columns: time, Temp, Humidity, WetBulb"
)

# Load and process data
@st.cache_data
def load_and_process_data(file):
    """Load and preprocess the data"""
    df = pd.read_csv(file)
    
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # Create additional time-based columns
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day_of_year'] = df['time'].dt.dayofyear
    df['hour'] = df['time'].dt.hour
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })
    
    # Additional calculations
    df['temp_wetbulb_diff'] = df['Temp'] - df['WetBulb']
    df['WetBulb_7day_avg'] = df['WetBulb'].rolling(window=24*7, center=True).mean()
    df['WetBulb_30day_avg'] = df['WetBulb'].rolling(window=24*30, center=True).mean()
    
    return df

# Main application logic
if uploaded_file is not None:
    # Load data
    with st.spinner('Loading and processing data...'):
        df = load_and_process_data(uploaded_file)
    
    st.success(f"✅ Data loaded successfully! {len(df):,} records from {df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}")
    
    # Sidebar controls
    st.sidebar.markdown("### 🎛️ Filter Controls")
    
    # Date range selector
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['time'].min().date(), df['time'].max().date()),
        min_value=df['time'].min().date(),
        max_value=df['time'].max().date()
    )
    
    # Year selector
    available_years = sorted(df['year'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Years",
        options=available_years,
        default=available_years,
        help="Choose specific years to analyze"
    )
    
    # Season selector
    selected_seasons = st.sidebar.multiselect(
        "Select Seasons",
        options=['Spring', 'Summer', 'Autumn', 'Winter'],
        default=['Spring', 'Summer', 'Autumn', 'Winter']
    )
    
    # Hour range selector
    hour_range = st.sidebar.slider(
        "Hour Range",
        min_value=0,
        max_value=23,
        value=(0, 23),
        help="Select hour range for analysis"
    )
    
    # Extreme threshold selector
    extreme_threshold = st.sidebar.slider(
        "Extreme Events Threshold (%)",
        min_value=90,
        max_value=99,
        value=95,
        help="Percentile threshold for extreme events"
    )
    
    # Apply filters
    filtered_df = df[
        (df['time'].dt.date >= date_range[0]) &
        (df['time'].dt.date <= date_range[1]) &
        (df['year'].isin(selected_years)) &
        (df['season'].isin(selected_seasons)) &
        (df['hour'] >= hour_range[0]) &
        (df['hour'] <= hour_range[1])
    ].copy()
    
    if len(filtered_df) == 0:
        st.error("❌ No data available for the selected filters. Please adjust your selection.")
        st.stop()
    
    # Dashboard layout
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📊 Total Records",
            value=f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
        )
    
    with col2:
        st.metric(
            label="🌡️ Avg Wet Bulb",
            value=f"{filtered_df['WetBulb'].mean():.1f}°C",
            delta=f"{filtered_df['WetBulb'].mean() - df['WetBulb'].mean():.1f}°C"
        )
    
    with col3:
        st.metric(
            label="🔥 Max Wet Bulb",
            value=f"{filtered_df['WetBulb'].max():.1f}°C",
            delta=f"{filtered_df['WetBulb'].max() - df['WetBulb'].max():.1f}°C"
        )
    
    with col4:
        st.metric(
            label="🧊 Min Wet Bulb",
            value=f"{filtered_df['WetBulb'].min():.1f}°C",
            delta=f"{filtered_df['WetBulb'].min() - df['WetBulb'].min():.1f}°C"
        )
    
    with col5:
        st.metric(
            label="📈 Std Deviation",
            value=f"{filtered_df['WetBulb'].std():.1f}°C",
            delta=f"{filtered_df['WetBulb'].std() - df['WetBulb'].std():.1f}°C"
        )
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Time Series", "🌤️ Seasonal Analysis", "🔥 Extreme Events", 
        "📊 Statistical Analysis", "📋 Data Summary"
    ])
    
    with tab1:
        st.markdown("### 📈 Interactive Time Series Analysis")
        
        # Time series visualization options
        col1, col2 = st.columns([3, 1])
        
        with col2:
            show_temp = st.checkbox("Show Temperature", value=True)
            show_humidity = st.checkbox("Show Humidity", value=True)
            show_moving_avg = st.checkbox("Show Moving Averages", value=False)
            aggregation = st.selectbox(
                "Data Aggregation",
                options=["Hourly", "Daily", "Weekly", "Monthly"],
                index=0
            )
        
        with col1:
            # Create time series plot
            fig_ts = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Wet Bulb Temperature', 'Temperature & Humidity'),
                vertical_spacing=0.1,
                specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
            )
            
            # Aggregate data based on selection
            if aggregation == "Daily":
                plot_df = filtered_df.groupby(filtered_df['time'].dt.date).mean().reset_index()
                plot_df['time'] = pd.to_datetime(plot_df['time'])
            elif aggregation == "Weekly":
                plot_df = filtered_df.groupby(pd.Grouper(key='time', freq='W')).mean().reset_index()
            elif aggregation == "Monthly":
                plot_df = filtered_df.groupby(pd.Grouper(key='time', freq='M')).mean().reset_index()
            else:
                plot_df = filtered_df.copy()
            
            # Wet Bulb Temperature
            fig_ts.add_trace(
                go.Scatter(x=plot_df['time'], y=plot_df['WetBulb'],
                          mode='lines', name='Wet Bulb Temperature',
                          line=dict(color='#FF6B6B', width=2)),
                row=1, col=1
            )
            
            if show_moving_avg and 'WetBulb_7day_avg' in plot_df.columns:
                fig_ts.add_trace(
                    go.Scatter(x=plot_df['time'], y=plot_df['WetBulb_7day_avg'],
                              mode='lines', name='7-day Average',
                              line=dict(color='#FF6B6B', width=1, dash='dash')),
                    row=1, col=1
                )
            
            # Temperature and Humidity
            if show_temp:
                fig_ts.add_trace(
                    go.Scatter(x=plot_df['time'], y=plot_df['Temp'],
                              mode='lines', name='Temperature',
                              line=dict(color='#4ECDC4', width=1.5)),
                    row=2, col=1
                )
            
            if show_humidity:
                fig_ts.add_trace(
                    go.Scatter(x=plot_df['time'], y=plot_df['Humidity'],
                              mode='lines', name='Humidity',
                              line=dict(color='#45B7D1', width=1.5),
                              yaxis='y3'),
                    row=2, col=1
                )
            
            fig_ts.update_layout(height=600, template='plotly_white')
            fig_ts.update_yaxes(title_text="Wet Bulb Temperature (°C)", row=1, col=1)
            fig_ts.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
            fig_ts.update_yaxes(title_text="Humidity (%)", secondary_y=True, row=2, col=1)
            
            st.plotly_chart(fig_ts, use_container_width=True)
    
    with tab2:
        st.markdown("### 🌤️ Seasonal and Temporal Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Seasonal boxplot
            fig_seasonal = px.box(
                filtered_df, x='season', y='WetBulb',
                title='Seasonal Distribution of Wet Bulb Temperature',
                color='season',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            )
            fig_seasonal.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig_seasonal, use_container_width=True)
            
            # Monthly trends
            monthly_avg = filtered_df.groupby('month')['WetBulb'].mean().reset_index()
            monthly_avg['month_name'] = monthly_avg['month'].map({
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            })
            
            fig_monthly = px.line(
                monthly_avg, x='month_name', y='WetBulb',
                title='Monthly Average Wet Bulb Temperature',
                markers=True
            )
            fig_monthly.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            # Hourly patterns
            hourly_avg = filtered_df.groupby('hour')['WetBulb'].mean().reset_index()
            fig_hourly = px.line(
                hourly_avg, x='hour', y='WetBulb',
                title='Average Wet Bulb Temperature by Hour',
                markers=True
            )
            fig_hourly.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Yearly comparison
            yearly_avg = filtered_df.groupby('year')['WetBulb'].mean().reset_index()
            fig_yearly = px.line(
                yearly_avg, x='year', y='WetBulb',
                title='Yearly Average Wet Bulb Temperature',
                markers=True
            )
            fig_yearly.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig_yearly, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔥 Extreme Events Analysis")
        
        # Calculate extreme thresholds
        wb_high = filtered_df['WetBulb'].quantile(extreme_threshold/100)
        wb_low = filtered_df['WetBulb'].quantile((100-extreme_threshold)/100)
        
        extreme_high = filtered_df[filtered_df['WetBulb'] >= wb_high]
        extreme_low = filtered_df[filtered_df['WetBulb'] <= wb_low]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔥 Extreme High Events", len(extreme_high))
        with col2:
            st.metric("🧊 Extreme Low Events", len(extreme_low))
        with col3:
            st.metric("📊 Threshold", f"{extreme_threshold}th percentile")
        
        # Extreme events plot
        fig_extreme = go.Figure()
        
        # Main time series
        fig_extreme.add_trace(
            go.Scatter(x=filtered_df['time'], y=filtered_df['WetBulb'],
                      mode='lines', name='Wet Bulb Temperature',
                      line=dict(color='#BDC3C7', width=1), opacity=0.7)
        )
        
        # Extreme events
        if len(extreme_high) > 0:
            fig_extreme.add_trace(
                go.Scatter(x=extreme_high['time'], y=extreme_high['WetBulb'],
                          mode='markers', name=f'Extreme High (≥{wb_high:.1f}°C)',
                          marker=dict(color='#E74C3C', size=8, symbol='triangle-up'))
            )
        
        if len(extreme_low) > 0:
            fig_extreme.add_trace(
                go.Scatter(x=extreme_low['time'], y=extreme_low['WetBulb'],
                          mode='markers', name=f'Extreme Low (≤{wb_low:.1f}°C)',
                          marker=dict(color='#3498DB', size=8, symbol='triangle-down'))
            )
        
        # Threshold lines
        fig_extreme.add_hline(y=wb_high, line_dash="dash", line_color="#E74C3C")
        fig_extreme.add_hline(y=wb_low, line_dash="dash", line_color="#3498DB")
        
        fig_extreme.update_layout(
            title='Extreme Wet Bulb Temperature Events',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_extreme, use_container_width=True)
        
        # Extreme events table
        if len(extreme_high) > 0 or len(extreme_low) > 0:
            st.markdown("#### 📋 Extreme Events Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if len(extreme_high) > 0:
                    st.markdown("**🔥 Extreme High Events**")
                    high_summary = extreme_high.nlargest(10, 'WetBulb')[['time', 'WetBulb', 'Temp', 'Humidity']]
                    st.dataframe(high_summary, use_container_width=True)
            
            with col2:
                if len(extreme_low) > 0:
                    st.markdown("**🧊 Extreme Low Events**")
                    low_summary = extreme_low.nsmallest(10, 'WetBulb')[['time', 'WetBulb', 'Temp', 'Humidity']]
                    st.dataframe(low_summary, use_container_width=True)
    
    with tab4:
        st.markdown("### 📊 Statistical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation matrix
            corr_matrix = filtered_df[['Temp', 'Humidity', 'WetBulb']].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(3),
                texttemplate="%{text}",
                textfont={"size": 14}
            ))
            
            fig_corr.update_layout(
                title='Correlation Matrix',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            # Distribution plot
            fig_dist = px.histogram(
                filtered_df, x='WetBulb',
                title='Wet Bulb Temperature Distribution',
                nbins=50,
                marginal='box'
            )
            fig_dist.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Statistical summary table
        st.markdown("#### 📈 Statistical Summary")
        
        summary_stats = filtered_df[['Temp', 'Humidity', 'WetBulb']].describe().round(2)
        st.dataframe(summary_stats, use_container_width=True)
    
    with tab5:
        st.markdown("### 📋 Data Summary and Export")
        
        # Data overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📊 Dataset Overview")
            st.write(f"**Total Records:** {len(filtered_df):,}")
            st.write(f"**Date Range:** {filtered_df['time'].min().strftime('%Y-%m-%d %H:%M')} to {filtered_df['time'].max().strftime('%Y-%m-%d %H:%M')}")
            st.write(f"**Years Covered:** {', '.join(map(str, sorted(filtered_df['year'].unique())))}")
            st.write(f"**Seasons:** {', '.join(filtered_df['season'].unique())}")
            
            # Key insights
            st.markdown("#### 🔍 Key Insights")
            max_wb_idx = filtered_df['WetBulb'].idxmax()
            min_wb_idx = filtered_df['WetBulb'].idxmin()
            
            st.write(f"**Highest Wet Bulb:** {filtered_df.loc[max_wb_idx, 'WetBulb']:.2f}°C on {filtered_df.loc[max_wb_idx, 'time'].strftime('%Y-%m-%d %H:%M')}")
            st.write(f"**Lowest Wet Bulb:** {filtered_df.loc[min_wb_idx, 'WetBulb']:.2f}°C on {filtered_df.loc[min_wb_idx, 'time'].strftime('%Y-%m-%d %H:%M')}")
            
            seasonal_stats = filtered_df.groupby('season')['WetBulb'].agg(['mean', 'std']).round(2)
            most_variable_season = seasonal_stats['std'].idxmax()
            st.write(f"**Most Variable Season:** {most_variable_season} (σ = {seasonal_stats.loc[most_variable_season, 'std']:.2f}°C)")
            
            corr_temp = filtered_df['WetBulb'].corr(filtered_df['Temp'])
            corr_humidity = filtered_df['WetBulb'].corr(filtered_df['Humidity'])
            st.write(f"**Correlation with Temperature:** {corr_temp:.3f}")
            st.write(f"**Correlation with Humidity:** {corr_humidity:.3f}")
        
        with col2:
            st.markdown("#### 💾 Export Data")
            
            # Data export options
            export_format = st.selectbox(
                "Export Format",
                options=["CSV", "Excel", "JSON"]
            )
            
            include_calculated = st.checkbox("Include Calculated Fields", value=True)
            
            if st.button("📥 Download Filtered Data"):
                if include_calculated:
                    export_df = filtered_df.copy()
                else:
                    export_df = filtered_df[['time', 'Temp', 'Humidity', 'WetBulb']].copy()
                
                if export_format == "CSV":
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"wetbulb_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                elif export_format == "Excel":
                    import io
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        export_df.to_excel(writer, sheet_name='WetBulb_Data', index=False)
                    st.download_button(
                        label="Download Excel",
                        data=output.getvalue(),
                        file_name=f"wetbulb_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:  # JSON
                    json_data = export_df.to_json(orient='records', date_format='iso')
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"wetbulb_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        # Raw data preview
        st.markdown("#### 👀 Data Preview")
        show_columns = st.multiselect(
            "Select columns to display",
            options=filtered_df.columns.tolist(),
            default=['time', 'Temp', 'Humidity', 'WetBulb']
        )
        
        if show_columns:
            st.dataframe(
                filtered_df[show_columns].head(1000),
                use_container_width=True,
                height=400
            )

else:
    # Welcome screen
    st.markdown("""
    ### 👋 Welcome to the Wet Bulb Temperature Dashboard!
    
    This interactive dashboard allows you to:
    
    - 📈 **Visualize time series** of wet bulb temperature data
    - 🌤️ **Analyze seasonal patterns** and trends
    - 🔥 **Identify extreme events** and outliers
    - 📊 **Explore statistical relationships** between variables
    - 💾 **Export filtered data** in multiple formats
    
    **To get started:**
    1. Upload your CSV file using the sidebar
    2. Use the filter controls to focus on specific time periods
    3. Explore different visualization tabs
    4. Export your analysis results
    
    **Required CSV format:**
    - `time`: Date/time column
    - `Temp`: Temperature values
    - `Humidity`: Humidity values  
    - `WetBulb`: Wet bulb temperature values
    """)
    
    # Sample data format
    st.markdown("#### 📋 Expected Data Format")
    sample_data = pd.DataFrame({
        'time': ['2020-01-01 00:00:00', '2020-01-01 01:00:00', '2020-01-01 02:00:00'],
        'Temp': [25.5, 26.0, 25.8],
        'Humidity': [65.2, 68.1, 66.5],
        'WetBulb': [20.1, 20.8, 20.4]
    })
    st.dataframe(sample_data, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666666;'>"
    "🌡️ Wet Bulb Temperature Dashboard | Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True
)