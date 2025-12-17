import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import datetime

# =========================================================
# Application Configuration
# =========================================================
st.set_page_config(
    page_title="Transformer Oil Sampling Analysis Trend Development",
    layout="wide"
)

# =========================================================
# Global Constants and Helper Functions
# =========================================================

# Hardcoded login credentials (for academic/demo purpose)
USERNAME = "ketam"
PASSWORD = "ketam123"

# List of DGA gas parameters expected in the uploaded CSV file
# (Must match column names exactly)
DGA_GASES = [
    "Hydrogen (H2)", "Methane (CH4)", "Ethane (C2H6)",
    "Ethylene (C2H4)", "Acetylene (C2H2)",
    "Carbon Monoxide (CO)", "Carbon Dioxide (CO2)"
]
@st.cache_resource
def run_prophet_forecast(df, gas_col, forecast_years=5):
    """
    Trains a Prophet time-series model and forecasts future gas concentration.
    Annual data frequency is assumed.
    """

    prophet_df = df[["Sampling Date", gas_col]].rename(
        columns={"Sampling Date": "ds", gas_col: "y"}
    )

    prophet_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    prophet_df.dropna(inplace=True)

    if len(prophet_df) < 2:
        return None, None

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95
    )

    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=forecast_years, freq="AS")
    forecast = model.predict(future)

    return model, forecast


# =========================================================
# 1. Login Interface
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Transformer Health Monitoring System")
    st.subheader("User Authentication")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

    st.stop()


# =========================================================
# 2. Main Dashboard
# =========================================================
st.title("⚡ Transformer DGA Trend and Predictive Health Assessment")
st.success("Login successful. System ready for analysis.")
st.markdown("---")

# Transformer identification
transformer_name = st.text_input("Transformer Name / ID", "T-XYZ-123")

# File upload section
st.subheader("Upload DGA CSV File")
file = st.file_uploader("Upload CSV File", type=["csv"])

if not file:
    st.info("Please upload a DGA CSV file to continue.")
    st.stop()


# =========================================================
# 3. Data Loading and Preprocessing
# =========================================================
try:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    if "Sampling Date" not in df.columns:
        st.error("The CSV file must contain a 'Sampling Date' column.")
        st.stop()

    df["Sampling Date"] = pd.to_datetime(df["Sampling Date"], errors="coerce")

    for col in df.columns:
        if col != "Sampling Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    st.info("Data successfully loaded.")
    st.dataframe(df.head())

except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()


# =========================================================
# 4. Reference Limit Input (Standards-Based)
# =========================================================
st.subheader("Reference Gas Limits (ppm)")
st.caption("Based on international / utility laboratory standards.")

if "ref_limits" not in st.session_state:
    st.session_state.ref_limits = {gas: 0 for gas in DGA_GASES}

available_gases = [g for g in DGA_GASES if g in df.columns]

cols_per_row = 4
for i in range(0, len(available_gases), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, gas in enumerate(available_gases[i:i + cols_per_row]):
        with cols[j]:
            st.session_state.ref_limits[gas] = st.number_input(
                f"{gas} Limit (ppm)",
                min_value=0,
                value=st.session_state.ref_limits.get(gas, 0),
                key=f"limit_{gas}"
            )

st.markdown("---")


# =========================================================
# 5. Trend Analysis and ML Forecasting
# =========================================================
st.header(f"DGA Analysis for Transformer: {transformer_name}")

selected_gas = st.selectbox("Select Gas Parameter", available_gases)

if selected_gas:
    limit_value = st.session_state.ref_limits.get(selected_gas, 0)

    st.subheader("Forecast Configuration")
    forecast_years = st.slider("Forecast Period (Years)", 1, 10, 5)

    model, forecast_results = run_prophet_forecast(df, selected_gas, forecast_years)

    if model and forecast_results is not None:

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["Sampling Date"],
            y=df[selected_gas],
            mode="lines+markers",
            name="Historical Data"
        ))

        last_date = df["Sampling Date"].max()
        future_data = forecast_results[forecast_results["ds"] >= last_date]

        fig.add_trace(go.Scatter(
            x=future_data["ds"],
            y=future_data["yhat"],
            mode="lines",
            name="ML Forecast"
        ))

        fig.add_trace(go.Scatter(
            x=pd.concat([future_data["ds"], future_data["ds"].iloc[::-1]]),
            y=pd.concat([future_data["yhat_upper"], future_data["yhat_lower"].iloc[::-1]]),
            fill="toself",
            name="Forecast Confidence Interval",
            hoverinfo="skip"
        ))

        if limit_value > 0:
            fig.add_hline(y=limit_value, line_dash="dash", line_color="orange",
                          annotation_text="Warning Limit")
            fig.add_hline(y=limit_value * 2, line_dash="dash", line_color="red",
                          annotation_text="Critical Limit")

        fig.update_layout(
            title=f"{selected_gas} Trend and {forecast_years}-Year Prediction",
            xaxis_title="Year",
            yaxis_title="Gas Concentration (ppm)",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)


        # =================================================
        # 6. AI Recommendation Output
        # =================================================
        st.subheader("AI-Based Recommendation")

        latest_value = df[selected_gas].iloc[-1]

        if pd.isna(latest_value):
            st.warning("Latest data point is invalid. Analysis incomplete.")
        elif limit_value > 0:
            critical_cross = future_data[future_data["yhat"] >= limit_value * 2]

            if not critical_cross.empty:
                year = critical_cross.iloc[0]["ds"].year
                lead_time = year - datetime.date.today().year

                st.error("Critical condition predicted.")
                st.markdown(
                    f"The **{selected_gas}** concentration is forecasted to exceed "
                    f"the critical threshold in **{year}**, providing approximately "
                    f"**{lead_time} years** for maintenance planning."
                )
            else:
                st.success("No critical condition predicted within forecast period.")
                st.markdown("Transformer condition is expected to remain stable.")

    else:
        st.error("Insufficient data to perform forecasting.")
