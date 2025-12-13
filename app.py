import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# -------------------------------------------
# SIMPLE LOGIN (HARDCODED USERNAME & PASSWORD)
# -------------------------------------------
USERNAME = "ketam"
PASSWORD = "ketam123"

st.title("Transformer Oil Sampling Analysis Trend Development")

# --- LOGIN STATE ---
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
            st.error("Incorrect username or password.")

    # 🚫 Stop app here if not logged in
    st.stop()

# -------------------------------------------
# AFTER LOGIN — MAIN DASHBOARD
# -------------------------------------------
st.success("Logged in successfully!")

# Transformer label
transformer_name = st.text_input("Transformer Name / ID", "Transformer A")

# ------------------- UPLOAD CSV -------------------
st.subheader("Upload DGA CSV File")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        st.write("Columns detected:", df.columns.tolist())

        # Check Sampling Date
        if "Sampling Date" not in df.columns:
            st.error("CSV must contain 'Sampling Date' column.")
            st.stop()

        # Convert Sampling Date
        df["Sampling Date"] = pd.to_datetime(df["Sampling Date"], errors="coerce")

        # Convert all other columns to numeric
        for col in df.columns:
            if col != "Sampling Date":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Show data
        st.write("### Raw Uploaded Data")
        st.dataframe(df)

        # Detect numeric columns
        numeric_cols = df.select_dtypes(include=["float", "int"]).columns

        if len(numeric_cols) == 0:
            st.error("No numeric DGA data found.")
            st.stop()

        # Select parameter
        selected_gas = st.selectbox("Select DGA parameter", numeric_cols)

        # Plot
        fig = px.line(
            df,
            x="Sampling Date",
            y=selected_gas,
            markers=True,
            title=f"{selected_gas} Trend for {transformer_name}"
        )
        st.plotly_chart(fig)

        # ---------------- TREND ANALYSIS ----------------
        st.subheader("Trend Analysis")
        clean = df[["Sampling Date", selected_gas]].dropna()

        if len(clean) > 1:
            slope = np.polyfit(range(len(clean)), clean[selected_gas], 1)[0]

            if slope > 0:
                st.warning("Increasing trend detected — condition may be worsening.")
            elif slope < 0:
                st.success("Decreasing trend — condition improving.")
            else:
                st.info("Stable trend observed.")
        else:
            st.info("Not enough data for trend analysis.")

        # ---------------- AI RECOMMENDATION ----------------
        st.subheader("AI Recommendation (Rule-Based)")
        last_value = clean[selected_gas].iloc[-1]

        if "Hydrogen" in selected_gas or "H2" in selected_gas:
            if last_value > 500:
                st.error("High Hydrogen: Possible partial discharge or overheating.")
            else:
                st.success("Hydrogen within acceptable range.")

        elif "Methane" in selected_gas or "CH4" in selected_gas:
            if last_value > 300:
                st.error("High Methane: Thermal fault suspected.")
            else:
                st.success("Methane level normal.")

        elif "TDCG" in selected_gas:
            if last_value > 720:
                st.error("Critical TDCG level — immediate maintenance required.")
            elif last_value > 300:
                st.warning("Elevated TDCG — monitor closely.")
            else:
                st.success("TDCG normal.")

        else:
            st.info("No predefined diagnostic rule for this parameter.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CSV file to begin analysis.")
