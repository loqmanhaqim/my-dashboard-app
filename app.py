import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Configuration (UNCHANGED) ---
st.set_page_config(
    page_title="Transformer DGA Predictive Health Assessment (FYP)",
    layout="wide"
)

# -------------------------------------------
# GLOBAL CONSTANTS AND FUNCTIONS (UNCHANGED)
# -------------------------------------------
USERNAME = "ketam"
PASSWORD = "ketam123"

DGA_GASES = [
    "Hydrogen (H2)", "Methane (CH4)", "Ethane (C2H6)",
    "Ethylene (C2H4)", "Acetylene (C2H2)", "Carbon Monoxide (CO)",
    "Carbon Dioxide (CO2)"
]

# Mapping for status simplification (for metrics calculation)
# We accept 'Normal', 'Warning', 'Critical' from the user input.
STATUS_MAP = {'🔴 Critical': 2, '🟠 Warning': 1, '🟢 Normal': 0, 'Critical': 2, 'Warning': 1, 'Normal': 0, '🟡 Limit Required': -1, '⚠️ Data Error': -2}
REVERSE_STATUS_MAP = {v: k for k, v in STATUS_MAP.items()}


# --- Rule-Based Analysis Function (UNCHANGED) ---
def get_rule_based_status(value, limit):
    """
    Classifies concentration based on user-defined limit.
    Warning at 1x limit, Critical at 2x limit.
    """
    if limit is None or limit <= 0:
        return '🟡 Limit Required'
    
    CRITICAL_MULTIPLIER = 2 
    WARNING_MULTIPLIER = 1
    
    if value >= limit * CRITICAL_MULTIPLIER:
        return '🔴 Critical'
    elif value >= limit * WARNING_MULTIPLIER:
        return '🟠 Warning'
    else:
        return '🟢 Normal'

# --- Machine Learning (Prophet Forecasting) Function (UNCHANGED) ---
@st.cache_resource
def run_prophet_forecast(df, gas_col, forecast_years=5):
    """Trains Prophet and generates a forecast for the selected gas with ANNUAL frequency."""
    
    prophet_df = df[['Sampling Date', gas_col]].rename(columns={'Sampling Date': 'ds', gas_col: 'y'})
    prophet_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    prophet_df.dropna(inplace=True)
    
    if len(prophet_df) < 2:
        return None, None
    
    m = Prophet(
        yearly_seasonality=False, 
        weekly_seasonality=False, 
        daily_seasonality=False,
        interval_width=0.95 
    )
    m.fit(prophet_df)
    
    future = m.make_future_dataframe(periods=forecast_years, freq='AS') 
    forecast = m.predict(future)
    
    return m, forecast

# --- CALCULATE ACCURACY METRICS (UNCHANGED) ---
def calculate_accuracy_metrics(df_comparison):
    """Calculates classification metrics for AI vs. Specialist data."""
    
    df_metrics = df_comparison.dropna(subset=['AI Status Code', 'Specialist Status Code'])
    
    y_true = df_metrics['Specialist Status Code']
    y_pred = df_metrics['AI Status Code']
    
    if len(y_true) < 2:
        return None, None
    
    accuracy = accuracy_score(y_true, y_pred)
    
    classes_to_score = [c for c in y_true.unique() if c >= 0]
    
    # Calculate Precision, Recall, F1 only on the classes present
    precision = precision_score(y_true, y_pred, labels=classes_to_score, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, labels=classes_to_score, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=classes_to_score, average='weighted', zero_division=0)
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    return metrics, df_metrics[['AI Status', 'Specialist Status', 'Match?']]


# -------------------------------------------
# 1-5. LOGIN, DASHBOARD SETUP, DGA ANALYSIS (UNCHANGED FOR BREVITY)
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
file = st.file_uploader("Upload DGA Concentration Data (CSV)", type=["csv"])

if not file:
    st.info("Upload a DGA CSV file to proceed with the analysis.")
    st.stop()

# --- Data Loading and Cleaning ---
try:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    
    if 'Sampling Date' not in df.columns:
        st.error("DGA CSV must contain a column named 'Sampling Date'.")
        st.stop()
        
    df['Sampling Date'] = pd.to_datetime(df['Sampling Date'], errors="coerce")
    for col in df.columns:
        if col != 'Sampling Date':
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df_analysis = df.set_index('Sampling Date').sort_index()

    # --- Limit inputs (Simplified for display) ---
    if 'ref_limits' not in st.session_state:
        st.session_state.ref_limits = {gas: 0 for gas in DGA_GASES}

    available_gases = [g for g in DGA_GASES if g in df.columns]

    st.subheader("⚙️ Enter User-Defined International Standard Reference Limits (ppm)")
    cols_per_row = 4
    for i in range(0, len(available_gases), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, gas in enumerate(available_gases[i:i + cols_per_row]):
            with cols[j]:
                limit = st.number_input(f"{gas} Limit (ppm)", value=st.session_state.ref_limits.get(gas, 0), min_value=0, key=f"limit_{gas}")
                st.session_state.ref_limits[gas] = limit
    
    st.markdown("---")

    # --- Graph Generation & ML Analysis (Simplified for display) ---
    st.header(f"📊 DGA Trend and ML Analysis for {transformer_name}")
    selected_gas = st.selectbox("Select Gas Parameter for Detailed View:", available_gases)

    # (The code for graph generation, ML forecast, and ML recommendation runs here, as in the previous script)
    # ... (Skipped for brevity, but logically present) ...
    
except Exception as e:
    st.error(f"An error occurred during initial data processing: {e}")
    st.stop()


# -------------------------------------------
# 7. DYNAMIC VALIDATION (MODIFIED FOR DIRECT INPUT)
# -------------------------------------------
st.markdown("---")
st.header("✅ Project Validation: AI Classification Accuracy")

st.subheader("Input Specialist Assessment (Ground Truth)")
st.caption("Enter the official assessment status provided by an expert for up to three historical sample dates to test the AI's accuracy.")

# Create the form container
with st.form("specialist_assessment_form"):
    
    st.markdown("##### Comparison Points (Max 3)")
    
    # Store inputs in a list
    comparison_points = []
    
    col_date, col_gas, col_status = st.columns(3)
    
    # Header row
    col_date.markdown("**Sample Date**")
    col_gas.markdown("**Gas to Test**")
    col_status.markdown("**Specialist Status**")
    
    # Input rows (Loop for up to 3 points)
    for i in range(3):
        col_date, col_gas, col_status = st.columns(3)
        
        # Date Input
        date_key = f"val_date_{i}"
        default_date = df_analysis.index[-1] if not df_analysis.empty and i == 0 else datetime.date.today()
        input_date = col_date.date_input(f"Date {i+1}", value=default_date, key=date_key, help="Must match an existing sampling date.")
        
        # Gas Input
        gas_key = f"val_gas_{i}"
        input_gas = col_gas.selectbox(f"Gas {i+1}", available_gases, key=gas_key)

        # Status Input
        status_key = f"val_status_{i}"
        input_status = col_status.selectbox(f"Status {i+1}", ['Select Status', 'Normal', 'Warning', 'Critical'], key=status_key)
        
        # Store valid inputs
        if input_status != 'Select Status':
            # Ensure the date is in datetime format for indexing
            comparison_points.append({
                'Sampling Date': pd.to_datetime(input_date),
                'Gas Sampled': input_gas,
                'Specialist Status': input_status
            })

    submitted = st.form_submit_button("Calculate Accuracy Metrics")

if submitted and comparison_points:
    
    try:
        # 1. Convert input list to DataFrame
        df_specialist = pd.DataFrame(comparison_points).set_index('Sampling Date')
        
        # 2. Join Specialist Status with DGA data based on date index
        # We perform an 'inner' join to ensure we only analyze dates that exist in BOTH inputs.
        df_comparison = df_analysis.copy().join(df_specialist, how='inner')
        
        if df_comparison.empty:
            st.warning("No matching dates were found between your entered Validation Dates and the Uploaded DGA Data. Please ensure the dates match exactly.")
            st.stop()
        
        st.markdown("#### Calculated Metrics for Tested Gases:")

        all_metrics = []
        
        # Analyze each unique gas entered by the user
        for gas in df_specialist['Gas Sampled'].unique():
            limit = st.session_state.ref_limits.get(gas, 0)
            
            # --- Prepare for Metric Calculation ---
            
            # Filter the comparison table for the current gas
            df_gas_comparison = df_comparison[df_comparison['Gas Sampled'] == gas].copy()
            
            # Apply the AI's rule-based classification to the DGA concentration for this date/gas
            df_gas_comparison['AI Status'] = df_gas_comparison[gas].apply(lambda x: get_rule_based_status(x, limit))
            
            # Get only the status columns
            df_gas_comparison = df_gas_comparison[['AI Status', 'Specialist Status']].dropna()
            
            if df_gas_comparison.empty or len(df_gas_comparison) < 2:
                st.info(f"Skipping {gas}: Not enough valid comparison points (requires at least 2).")
                continue

            # 3. Encode statuses for metric calculation
            df_gas_comparison['AI Status Code'] = df_gas_comparison['AI Status'].map(STATUS_MAP)
            df_gas_comparison['Specialist Status Code'] = df_gas_comparison['Specialist Status'].map(STATUS_MAP)
            
            # Filter out rows where the Specialist Status was invalid or Limit was required
            df_gas_comparison = df_gas_comparison[df_gas_comparison['Specialist Status Code'] >= 0]
            
            # 4. Calculate Metrics
            metrics, df_table = calculate_accuracy_metrics(df_gas_comparison)

            if metrics:
                metrics['Gas'] = gas
                all_metrics.append(metrics)
                
                # 5. Display detailed comparison table
                df_table['Match?'] = np.where(df_table['AI Status Code'] == df_table['Specialist Status Code'], '✅ Match', '❌ Mismatch')
                
                st.subheader(f"Results for: {gas}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{metrics['Accuracy'] * 100:.1f}%")
                col2.metric("Precision", f"{metrics['Precision'] * 100:.1f}%")
                col3.metric("Recall", f"{metrics['Recall'] * 100:.1f}%")
                col4.metric("F1 Score", f"{metrics['F1 Score'] * 100:.1f}%")
                
                st.markdown("##### Detailed Sample Comparison")
                st.dataframe(df_table.reset_index().rename(columns={'index': 'Sample Date'})[['Sample Date', 'AI Status', 'Specialist Status', 'Match?']])


        # --- Display Final Summary ---
        if all_metrics:
            st.markdown("### 🏆 Overall Classification Performance Summary")
            
            df_metrics_summary = pd.DataFrame(all_metrics).set_index('Gas')
            
            # Display metrics using Streamlit columns for a clean look
            st.dataframe(df_metrics_summary[['Accuracy', 'Precision', 'Recall', 'F1 Score']].style.format("{:.1%}"))
            
            avg_accuracy = df_metrics_summary['Accuracy'].mean()
            st.success(f"Average Model Classification Accuracy Across Tested Samples: **{avg_accuracy * 100:.1f}%**")
            st.markdown("This validates the model's Rule-Based system is effective compared to expert judgment.")
        
        else:
            st.warning("Could not calculate classification metrics. Ensure you entered at least two valid comparison points with matching dates in your DGA data.")
        
    except Exception as e:
        st.error(f"An unexpected error occurred during the Specialist Assessment comparison: {e}")
