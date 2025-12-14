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

st.title("Transformer Oil DGA Trend & ML Health Assessment")

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

        # ----------------- ML FEATURE ENGINEERING -----------------
        features = []
        labels = []

        for _, row in df.iterrows():
            severity = 0  # Normal
            row_features = []

            for gas, limit in REFERENCE_LIMITS.items():
                value = row.get(gas, np.nan)

                # Safe numeric check
                if isinstance(value, (int, float)) and not np.isnan(value):
                    ratio = value / limit
                    row_features.append(ratio)
                    if ratio > 1:
                        severity = 2  # Critical
                    elif ratio > 0.5 and severity < 1:
                        severity = 1  # Warning
                else:
                    row_features.append(0)

            features.append(row_features)
            labels.append(severity)

        X = np.array(features)
        y = np.array(labels)

        # ----------------- TRAIN ML MODEL -----------------
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X, y)

        # ----------------- TREND GRAPH -----------------
        available_gases = [gas for gas in REFERENCE_LIMITS.keys() if gas in df.columns]
        if available_gases:
            selected_gas = st.selectbox("Select Gas for Trend", available_gases)

            fig = px.line(df, x="Sampling Date", y=selected_gas, markers=True,
                          title=f"{selected_gas} Trend - {transformer_name}")
            st.plotly_chart(fig)
        else:
            st.warning("No DGA gas columns found in CSV for trend plotting.")

        # ----------------- ML PREDICTION -----------------
        if len(X) > 0:
            latest = X[-1].reshape(1, -1)
            prediction = model.predict(latest)[0]

            st.subheader("ML-Based Transformer Condition")

            if prediction == 0:
                st.success("Normal Condition")
                st.write("Gas levels are within acceptable IEC/IEEE limits. Continue routine monitoring.")

            elif prediction == 1:
                st.warning("Warning Condition")
                st.write("Some gases are approaching standard limits. Increase monitoring frequency and plan inspection.")

            else:
                st.error("Critical Condition")
                st.write("One or more gases exceed IEC/IEEE limits. Immediate investigation and maintenance recommended.")
        else:
            st.info("Not enough data to predict transformer condition.")

    except Exception as e:
        st.error(f"Error reading CSV file: {e}")

else:
    st.info("Upload a DGA CSV file to start analysis")
