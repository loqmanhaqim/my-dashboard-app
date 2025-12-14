import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import datetime

# --- Configuration ---
st.set_page_config(
    page_title="Transformer DGA Predictive Health Assessment (FYP)",
    layout="wide"
)

# -------------------------------------------
# GLOBAL CONSTANTS AND FUNCTIONS
# -------------------------------------------
USERNAME = "ketam"
PASSWORD = "ketam123"

# List of DGA gases to analyze (Ensure these match your CSV column headers exactly)
DGA_GASES = [
    "Hydrogen (H2)", "Methane (CH4)", "Ethane (C2H6)",
    "Ethylene (C2H4)", "Acetylene (C2H2)", "Carbon Monoxide (CO)",
    "Carbon Dioxide (CO2)"
]

# --- Rule-Based Analysis Function (FIXED) ---
def get_rule_based_status(value, limit):
    """
    Classifies concentration based on user-defined limit.
    Warning at 1x limit, Critical at 2x limit.
    """
    # 1. Handle case where limit is not defined or zero
    if limit is None or limit <= 0:
        return '🟡 Limit Required'
    
    CRITICAL_MULTIPLIER = 2 
    WARNING_MULTIPLIER = 1
    
    # 2. Check Critical first
    if value >= limit * CRITICAL_MULTIPLIER:
        return '🔴 Critical'
    # 3. Check Warning
    elif value >= limit * WARNING_MULTIPLIER:
        return '🟠 Warning'
    # 4. Otherwise, Normal
    else:
        return '🟢 Normal'

# --- Machine Learning (Prophet Forecasting) Function ---
@st.cache_resource
def run_prophet_forecast(df, gas_col, forecast_years=5):
    """Trains Prophet and generates a forecast for the selected gas with ANNUAL frequency."""
    
    # 1. Prepare data for Prophet: needs columns named 'ds' and 'y'
    prophet_df = df[['Sampling Date', gas_col]].rename(columns={'Sampling Date': 'ds', gas_col: 'y'})
    
    # Clean data
    prophet_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    prophet_df.dropna(inplace=True)
    
    if len(prophet_df) < 2:
        return None, None
    
    # 2. Train Prophet model
    m = Prophet(
        yearly_seasonality=False, 
        weekly_seasonality=False, 
        daily_seasonality=False,
        interval_width=0.95 
    )
    m.fit(prophet_df)
    
    # 3. Create future dataframe and generate forecast
    # VITAL: Use 'AS' (Annual Start) frequency for one data point per year
    future = m.make_future_dataframe(periods=forecast_years, freq='AS') 
    forecast = m.predict(future)
    
    return m, forecast


# -------------------------------------------
# 1. LOGIN INTERFACE
# -------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Transformer Health Monitoring - Login")
    st.subheader("Login Required")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password")
    st.stop()

# -------------------------------------------
# 2. DASHBOARD / ANALYSIS STAGE
# -------------------------------------------
st.title("⚡ Transformer DGA Trend & Predictive Health Assessment")
st.success("Logged in successfully. Proceed to analysis.")
st.markdown("---")

# 3. User Input (Transformer Name & Data Upload)
transformer_name = st.text_input("Enter Transformer Name / ID:", value="T-XYZ-123")
st.subheader("Upload DGA CSV File")
file = st.file_uploader("Upload CSV File", type=["csv"])

if not file:
    st.info("Upload a DGA CSV file to proceed with the analysis. Ensure the file has a 'Sampling Date' column and gas columns.")
    st.stop()

# --- Data Loading and Cleaning ---
try:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    
    # Standardize the date column name
    if 'Sampling Date' not in df.columns:
        st.error("CSV must contain a column named 'Sampling Date'. Please rename the column in your file.")
        st.stop()
        
    df['Sampling Date'] = pd.to_datetime(df['Sampling Date'], errors="coerce")
    
    # Convert all other columns to numeric
    for col in df.columns:
        if col != 'Sampling Date':
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    st.info("Data Loaded Successfully. First 5 rows:")
    st.dataframe(df.head())

except Exception as e:
    st.error(f"Error reading or processing file. Please check format and column names: {e}")
    st.stop()


# 4. User-Defined Reference Limits
st.subheader("⚙️ Enter User-Defined International Standard Reference Limits (ppm)")
st.caption("These limits define the 'Warning' threshold (1x limit) and the 'Critical' threshold (2x limit) for analysis.")

if 'ref_limits' not in st.session_state:
    st.session_state.ref_limits = {gas: 0 for gas in DGA_GASES}

# Layout for inputs: show only gases present in the uploaded file
cols_per_row = 4
available_gases = [g for g in DGA_GASES if g in df.columns]

for i in range(0, len(available_gases), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, gas in enumerate(available_gases[i:i + cols_per_row]):
        with cols[j]:
            # Use get() for safe access to session state dictionary
            limit = st.number_input(f"{gas} Limit (ppm)", 
                                    value=st.session_state.ref_limits.get(gas, 0), 
                                    min_value=0, 
                                    key=f"limit_{gas}")
            st.session_state.ref_limits[gas] = limit
    
st.markdown("---")

# 5. Analysis & Graph Generation
st.header(f"📊 DGA Trend and ML Analysis for {transformer_name}")

selected_gas = st.selectbox("Select Gas Parameter for Detailed View:", available_gases)

if selected_gas:
    # --- PULL THE LIMIT FROM SESSION STATE ---
    limit_value = st.session_state.ref_limits.get(selected_gas, 0)
    
    # --- ML Forecasting Slider (Now in Years) ---
    st.subheader(f"Predictive Model Configuration: {selected_gas}")
    forecast_years = st.slider("Select Forecasting Period (Years):", 1, 10, 5)
    
    # Run ML Forecasting
    model, forecast_results = run_prophet_forecast(df, selected_gas, forecast_years)
    
    if model and forecast_results is not None:
        
        # --- Generate Plotly Graph ---
        fig = go.Figure()

        # 1. Historical Data
        fig.add_trace(go.Scatter(x=df['Sampling Date'], y=df[selected_gas], mode='lines+markers', name='Historical Data', line=dict(color='blue')))
        
        # 2. Forecast Data (Prophet yhat) - Start plotting forecast from the end of historical data
        historical_end_date = df['Sampling Date'].max()
        future_forecast_results = forecast_results[forecast_results['ds'] >= historical_end_date]
        
        fig.add_trace(go.Scatter(x=future_forecast_results['ds'], y=future_forecast_results['yhat'], 
                                 mode='lines', name='ML Forecast (Mean)', line=dict(color='green', dash='dot')))
        
        # 3. Confidence Interval (Uncertainty) - Only for the future period
        fig.add_trace(go.Scatter(
            x=pd.concat([future_forecast_results['ds'], future_forecast_results['ds'].iloc[::-1]]),
            y=pd.concat([future_forecast_results['yhat_upper'], future_forecast_results['yhat_lower'].iloc[::-1]]),
            fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", name='Uncertainty Interval'
        ))

        # 4. User-Defined Reference Limits (Rule-Based Overlays)
        critical_limit_val = limit_value * 2
        if limit_value > 0:
            
            # Warning (1x Limit)
            fig.add_hline(y=limit_value, line_dash="dash", line_color="orange", 
                          annotation_text=f"Warning Limit ({limit_value} ppm)", annotation_position="top left")
            # Critical (2x Limit)
            fig.add_hline(y=critical_limit_val, line_dash="dash", line_color="red", 
                          annotation_text=f"Critical Limit ({critical_limit_val} ppm)", annotation_position="bottom right")

        fig.update_layout(
            title=f"{selected_gas} Concentration Trend & {forecast_years}-Year ML Forecast for {transformer_name}",
            xaxis_title="Sampling Date (Annual Data)",
            yaxis_title=f"{selected_gas} Concentration (ppm)",
            hovermode="x unified",
            height=550
        )
        st.plotly_chart(fig, use_container_width=True)


        # --- 6. ML & Rule-Based Conclusion/Recommendation ---
st.subheader("💡 Analysis and AI Recommendation")

# FIX: Use .iloc[-1] to get the last value, then use .item() to pull the scalar value.
# Also, explicitly check if the value is NaN before proceeding with the classification.
latest_value_raw = df[selected_gas].iloc[-1]

# Check for NaN and convert to a comparable number (or handle explicitly)
if pd.isna(latest_value_raw):
    latest_value = -1 # Use a negative number so it always triggers 'Limit Required' logic in the function
    current_status = '⚠️ Data Error'
    limit_value = st.session_state.ref_limits.get(selected_gas, 0)
else:
    latest_value = latest_value_raw
    limit_value = st.session_state.ref_limits.get(selected_gas, 0)
    current_status = get_rule_based_status(latest_value, limit_value)


# A. Rule-Based Status (Latest Sample)
# ...
# Update the st.metric line to show the raw value correctly
st.metric(f"Current {selected_gas} Status (Latest Sample: {latest_value_raw} ppm)", current_status, delta_color=delta_color)

        # B. ML Predictive Warning
        if limit_value > 0:
            # Check if the forecast crosses the critical limit (yhat = mean prediction)
            forecast_critical = future_forecast_results[future_forecast_results['yhat'] >= critical_limit_val]
            
            if not forecast_critical.empty:
                projection_entry = forecast_critical.iloc[0]
                projection_date = projection_entry['ds'].strftime('%Y-%m-%d')
                lead_time_years = projection_entry['ds'].year - datetime.date.today().year
                
                st.error(f"**ML PREDICTIVE WARNING:** Critical Threshold Breach Projected!")
                st.markdown(f"The **{selected_gas}** concentration is forecasted to reach the Critical Limit of **{critical_limit_val} ppm** on or around **{projection_date}**.")
                st.markdown(f"This provides a lead time of approximately **{lead_time_years} years** for maintenance planning.")
                
            else:
                st.success("ML Forecast: Trend is projected to remain below the Critical Limit within the selected prediction window.")
                st.markdown("Recommendation: Maintain annual sampling frequency. Trend is stable based on historical data.")
                
        elif "Limit Required" in current_status:
             st.warning("Cannot provide predictive warning. Please define a positive Reference Limit to enable Rule-Based and ML analysis.")
    
    elif model is None and forecast_results is None:
        st.error("Not enough data points (requires at least two annual samples) to run the ML forecast for this gas.")
