import streamlit as st
import pandas as pd
import plotly.express as px

st.title("CSV Dashboard Example")

# Login Section
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if username == "admin" and password == "1234":
    st.success("Login Successful!")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Data Table")
        st.dataframe(df)

        column_options = df.columns.tolist()
        
        x_axis = st.selectbox("Select X-axis", column_options)
        y_axis = st.selectbox("Select Y-axis", column_options)

        fig = px.line(df, x=x_axis, y=y_axis, title="Generated Graph")
        st.plotly_chart(fig)

else:
    st.warning("Please login to continue.")
