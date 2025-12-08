import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st


# -------------------------------------------
# SIMPLE LOGIN (HARDCODED USERNAME & PASSWORD)
# -------------------------------------------
USERNAME = "admin"
PASSWORD = "1234"


st.title("Transformer Oil DGA Dashboard")


# ---- LOGIN FORM ----
if "logged_in" not in st.session_state:
st.session_state.logged_in = False


if not st.session_state.logged_in:
st.subheader("Login Required")
user = st.text_input("Username")
pwd = st.text_input("Password", type="password")


if st.button("Login"):
if user == USERNAME and pwd == PASSWORD:
st.session_state.logged_in = True
else:
st.error("Incorrect username or password.")
st.stop()


# -------------------------------------------
# AFTER LOGIN — MAIN DASHBOARD
# -------------------------------------------
st.success("Logged in successfully!")


# Transformer label
transformer_name = st.text_input("Transformer Name / ID", "Transformer A")


# Upload CSV
st.subheader("Upload DGA CSV File")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])


if uploaded_file:
df = pd.read_csv(uploaded_file)


# Try convert sampling date
if "Sampling Date" in df.columns:
try:
df["Sampling Date"] = pd.to_datetime(df["Sampling Date"], errors="coerce")
except:
st.warning("Could not convert sampling dates.")


st.write("### Raw Uploaded Data")
st.dataframe(df)


# ------------------ GRAPH ------------------
st.write("### Trend Graph")
numeric_cols = df.select_dtypes(include=["float", "int"]).columns


selected_gas = st.selectbox("Select parameter to plot", numeric_cols)


fig = px.line(df, x="Sampling Date", y=selected_gas, markers=True,
title=f"Trend of {selected_gas} for {transformer_name}")
st.plotly_chart(fig)


# ------------------ TREND ANALYSIS ------------------
st.write("### Trend Analysis")
clean = df[["Sampling Date", selected_gas]].dropna()
if len(clean) > 1:
slope = np.polyfit(range(len(clean)), clean[selected_gas], 1)[0]
if slope > 0:
st.info(f"**{selected_gas} is increasing over time.** Potential fault worsening.")
elif slope < 0:
st.info(f"**{selected_gas} is decreasing over time.** Condition improving.")
else:
st.info(f"**{selected_gas} is stable.** No major trend.")
else:
st.write("Not enough data for trend analysis.")


# ------------------ AI RECOMMENDATION ------------------
st.write("### AI Recommendation (Simple Rule-Based)")
last_value = clean[selected_gas].iloc[-1]


if "Hydrogen" in selected_gas or "H2" in selected_gas:
if last_value > 500:
st.error("High Hydrogen detected — possible partial discharge or overheating.")
else:
st.success("Hydrogen levels normal.")


elif "Methane" in selected_gas or "CH4" in selected_gas:
if last_value > 300:
st.error("High Methane — overheating risk.")
else:
st.success("Methane levels normal.")


elif "TDCG" in selected_gas:
if last_value > 720:
st.error("TDCG very high — urgent maintenance required.")
elif last_value > 300:
st.warning("TDCG moderately high — monitor closely.")
else:
st.success("TDCG normal.")


else:
st.info("No predefined rule for this parameter yet.")


else:
st.info("Please upload your DGA CSV file to begin analysis.")
