# --- Machine Learning (Prophet Forecasting) Function ---
@st.cache_resource
def run_prophet_forecast(df, gas_col, forecast_periods=5): # Defaulting to 5 years forecast
    """Trains Prophet and generates a forecast for the selected gas."""
    
    # 1. Prepare data for Prophet: needs columns named 'ds' and 'y'
    prophet_df = df[['Sampling Date', gas_col]].rename(columns={'Sampling Date': 'ds', gas_col: 'y'})
    
    # 2. Train Prophet model
    m = Prophet(
        yearly_seasonality=False, # Must be False for annual data
        weekly_seasonality=False, 
        daily_seasonality=False,
        interval_width=0.95 
    )
    m.fit(prophet_df)
    
    # 3. Create future dataframe and generate forecast
    # *** VITAL CHANGE: Use 'AS' (Annual Start) frequency ***
    future = m.make_future_dataframe(periods=forecast_periods, freq='AS') 
    forecast = m.predict(future)
    
    return m, forecast

# ... In the Streamlit body:
# Change the slider for consistency
forecast_years = st.slider("Select Forecasting Period (Years):", 1, 10, 5) # Changed months to years
model, forecast_results = run_prophet_forecast(df, selected_gas, forecast_years)
