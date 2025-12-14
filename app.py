import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

# -------------------------------------------
# LOGIN (HARDCODED)
# -------------------------------------------
USERNAME = "ketam"
PASSWORD = "ketam123"

st.title("Transformer Oil DGA Trend & Health Assessment")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
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
# DASHBOARD
# -------------------------------------------
st.success("Logged in successfully")
transformer_name = st.text_input("Transformer Name / ID")

# Reference limits (IEC / IEEE based)
REFERENCE_LIMITS = {
    "Hydrogen (H2)": 100,
    "Methane (CH4)": 120,
    "Carbon Monoxide (CO)": 350,
    "Carbon Dioxide (CO2)": 2500,
    "Ethylene (C2H4)": 50,
    "Ethane (C2H6)": 65,
    "Acetylene (C2H2)": 1,
}

st.subheader("Upload DGA CSV File")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        if "Sampling Date" not in df.columns:
            st.error("Sampling Date column is required in CSV")
            st.stop()

        # Convert date and numeric columns safely
        df["Sampling Date"] = pd.to_datetime(df["Sampling Date"], errors="coerce")
        for col in df.columns:
            if col != "Sampling Date":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        st.dataframe(df)

        # ----------------- TREND GRAPH -----------------
        available_gases = [gas for gas in REFERENCE_LIMITS.keys() if gas in df.columns]
        if available_gases:
            selected_gas = st.selectbox("Select Gas for Trend", available_gases)

            fig = px.line(df, x="Sampling Date", y=selected_gas, markers=True,
                          title=f"{selected_gas} Trend - {transformer_name}")
            st.plotly_chart(fig)
        else:
            st.warning("No DGA gas columns found in CSV for trend plotting.")

        # ----------------- DIRECT DGA-BASED CONDITION ASSESSMENT -----------------
        if len(df) > 0:
            latest_row = df.iloc[-1]
            severity = 0
            messages = []

            for gas, limit in REFERENCE_LIMITS.items():
                value = latest_row.get(gas, np.nan)
                if isinstance(value, (int, float)) and not np.isnan(value):
                    ratio = value / limit
                    if ratio > 1:
                        severity = max(severity, 2)
                        messages.append(f"{gas} is above limit! Possible critical condition (e.g., high energy arcing for acetylene).")
                    elif ratio > 0.5:
                        severity = max(severity, 1)
                        messages.append(f"{gas} is approaching limit. Warning condition.")

            st.subheader("Transformer Condition Based on Latest DGA Reading")

            if severity == 0:
                st.success("Normal Condition")
                st.write("Gas levels are within acceptable IEC/IEEE limits. Continue routine monitoring.")
            elif severity == 1:
                st.warning("Warning Condition")
                for msg in messages:
                    st.write(msg)
            else:
                st.error("Critical Condition")
                for msg in messages:
                    st.write(msg)
        else:
            st.info("Not enough data to assess transformer condition.")

    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

else:
    st.info("Upload a DGA CSV file to start analysis")

