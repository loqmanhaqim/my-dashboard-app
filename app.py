import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
import joblib
import datetime

# =========================================================
# App Configuration
# =========================================================
st.set_page_config(
    page_title="Transformer DGA ML Health Assessment",
    layout="wide"
)

# =========================================================
# Constants
# =========================================================
USERNAME = "ketam"
PASSWORD = "ketam123"

ML_FEATURES = [
    "Hydrogen (H2)",
    "Methane (CH4)",
    "Carbon Monoxide (CO)",
    "Carbon Dioxide (CO2)",
    "Ethylene (C2H4)",
    "Ethane (C2H6)",
    "Acetylene (C2H2)"
]

LABEL_MAP = {
    0: "🟢 Normal",
    1: "🟠 Warning",
    2: "🔴 Critical"
}

# =========================================================
# Load ML Model
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("dga_model.pkl")

clf_model = load_model()

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
        daily_seasonality=False
    )

    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_years, freq="AS")
    forecast = model.predict(future)

    return model, forecast

# =========================================================
# Login
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Transformer Health Monitoring System")
    st.subheader("Login")

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
# Dashboard
# =========================================================
st.title("⚡ Transformer DGA Trend & ML Health Assessment")
st.success("Login successful")

transformer_name = st.text_input("Transformer Name / ID", "T-XYZ-123")

st.subheader("Upload DGA CSV File")
file = st.file_uploader("Upload CSV", type=["csv"])

if not file:
    st.stop()

# =========================================================
# Load & Clean Data
# =========================================================
df = pd.read_csv(file)
df.columns = df.columns.str.strip()

if "Sampling Date" not in df.columns:
    st.error("CSV must contain 'Sampling Date'")
    st.stop()

df["Sampling Date"] = pd.to_datetime(df["Sampling Date"], errors="coerce")

for col in df.columns:
    if col != "Sampling Date":
        df[col] = pd.to_numeric(df[col], errors="coerce")

st.subheader("Data Preview")
st.dataframe(df.head())

# =========================================================
# Trend & Forecast
# =========================================================
available_gases = [g for g in ML_FEATURES if g in df.columns]
selected_gas = st.selectbox("Select Gas for Trend Analysis", available_gases)

if selected_gas:
    forecast_years = st.slider("Forecast Period (Years)", 1, 10, 5)

    model, forecast = run_prophet_forecast(df, selected_gas, forecast_years)

    if model:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["Sampling Date"],
            y=df[selected_gas],
            mode="lines+markers",
            name="Historical Data"
        ))

        future_data = forecast[forecast["ds"] >= df["Sampling Date"].max()]

        fig.add_trace(go.Scatter(
            x=future_data["ds"],
            y=future_data["yhat"],
            mode="lines",
            name="Forecast"
        ))

        fig.add_trace(go.Scatter(
            x=pd.concat([future_data["ds"], future_data["ds"].iloc[::-1]]),
            y=pd.concat([future_data["yhat_upper"], future_data["yhat_lower"].iloc[::-1]]),
            fill="toself",
            name="Confidence Interval",
            hoverinfo="skip"
        ))

        fig.update_layout(
            title=f"{selected_gas} Trend & Forecast",
            xaxis_title="Year",
            yaxis_title="ppm",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# ML-ONLY CONDITION ASSESSMENT
# =========================================================
st.subheader("🤖 Machine Learning Condition Assessment")

missing = [f for f in ML_FEATURES if f not in df.columns]

if missing:
    st.warning(f"Missing columns for ML prediction: {missing}")
else:
    latest_sample = df[ML_FEATURES].iloc[-1:].values
    prediction = clf_model.predict(latest_sample)[0]

    st.success(f"ML Predicted Condition: **{LABEL_MAP[prediction]}**")

    if prediction == 0:
        st.markdown(
            "The ML model predicts **Normal condition**, based on learned patterns "
            "from historical transformer data."
        )
    elif prediction == 1:
        st.markdown(
            "The ML model predicts a **Warning condition**, indicating abnormal "
            "gas behaviour compared to historical normal cases."
        )
    elif prediction == 2:
        st.markdown(
            "The ML model predicts a **Critical condition**, showing strong similarity "
            "to previously observed fault cases in the training data."
        )
