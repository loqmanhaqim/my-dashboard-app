import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
import joblib
import datetime

# =========================================================
# Application Configuration
# =========================================================
st.set_page_config(
    page_title="Transformer Oil Sampling Analysis Trend Development",
    layout="wide"
)

# =========================================================
# Global Constants
# =========================================================
USERNAME = "ketam"
PASSWORD = "ketam123"

DGA_GASES = [
    "Hydrogen (H2)", "Methane (CH4)", "Ethane (C2H6)",
    "Ethylene (C2H4)", "Acetylene (C2H2)",
    "Carbon Monoxide (CO)", "Carbon Dioxide (CO2)"
]

ML_FEATURES = [
    "Hydrogen (H2)",
    "Methane (CH4)",
    "Carbon Monoxide (CO)",
    "Carbon Dioxide (CO2)",
    "Ethylene (C2H4)",
    "Ethane (C2H6)",
    "Acetylene (C2H2)"
]

# =========================================================
# Load Supervised ML Model
# =========================================================
@st.cache_resource
def load_classification_model():
    return joblib.load("dga_model.pkl")

clf_model = load_classification_model()

# =========================================================
# Prophet Forecast Function
# =========================================================
@st.cache_resource
def run_prophet_forecast(df, gas_col, forecast_years=5):

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

transformer_name = st.text_input("Transformer Name / ID", "T-XYZ-123")

st.subheader("Upload DGA CSV File")
file = st.file_uploader("Upload CSV File", type=["csv"])

if not file:
    st.info("Please upload a DGA CSV file to continue.")
    st.stop()


# =========================================================
# 3. Data Loading and Preprocessing
# =========================================================
df = pd.read_csv(file)
df.columns = df.columns.str.strip()

if "Sampling Date" not in df.columns:
    st.error("The CSV file must contain a 'Sampling Date' column.")
    st.stop()

df["Sampling Date"] = pd.to_datetime(df["Sampling Date"], errors="coerce")

for col in df.columns:
    if col != "Sampling Date":
        df[col] = pd.to_numeric(df[col], errors="coerce")

st.subheader("Uploaded Data Preview")
st.dataframe(df.head())

available_gases = [g for g in DGA_GASES if g in df.columns]

# =========================================================
# 4. Reference Limits Input
# =========================================================
st.subheader("Reference Gas Limits (ppm)")

if "ref_limits" not in st.session_state:
    st.session_state.ref_limits = {gas: 0 for gas in available_gases}

cols = st.columns(4)
for i, gas in enumerate(available_gases):
    with cols[i % 4]:
        st.session_state.ref_limits[gas] = st.number_input(
            f"{gas} Limit (ppm)",
            min_value=0,
            value=st.session_state.ref_limits.get(gas, 0)
        )

st.markdown("---")


# =========================================================
# 5. Trend Analysis & Forecasting
# =========================================================
st.header(f"DGA Analysis for Transformer: {transformer_name}")

selected_gas = st.selectbox("Select Gas Parameter", available_gases)

if selected_gas:

    limit_value = st.session_state.ref_limits.get(selected_gas, 0)
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

        future_data = forecast_results[forecast_results["ds"] >= df["Sampling Date"].max()]

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
            name="Confidence Interval",
            hoverinfo="skip"
        ))

        if limit_value > 0:
            fig.add_hline(y=limit_value, line_dash="dash", line_color="orange",
                          annotation_text="Warning Limit")
            fig.add_hline(y=limit_value * 2, line_dash="dash", line_color="red",
                          annotation_text="Critical Limit")

        fig.update_layout(
            title=f"{selected_gas} Trend & {forecast_years}-Year Forecast",
            xaxis_title="Year",
            yaxis_title="Gas Concentration (ppm)",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 6. Supervised ML Classification
# =========================================================
st.subheader("🤖 Supervised ML Transformer Condition Classification")

missing = [f for f in ML_FEATURES if f not in df.columns]

if missing:
    st.warning(f"ML classification unavailable. Missing columns: {missing}")
else:
    latest_sample = df[ML_FEATURES].iloc[-1:].values
    prediction = clf_model.predict(latest_sample)[0]

    label_map = {0: "🟢 Normal", 1: "🟠 Warning", 2: "🔴 Critical"}

    st.success(f"Predicted Transformer Condition: **{label_map[prediction]}**")

    if prediction == 0:
        st.markdown(
            "The transformer is classified as **Normal**. "
            "Gas concentrations are within acceptable limits based on trained historical data."
        )
    elif prediction == 1:
        st.markdown(
            "The transformer is classified as **Warning**. "
            "At least one gas parameter exceeds reference limits. "
            "Closer monitoring is recommended."
        )
    else:
        st.markdown(
            "The transformer is classified as **Critical**. "
            "Multiple gas parameters exceed critical thresholds. "
            "Immediate inspection and maintenance action are strongly recommended."
        )
