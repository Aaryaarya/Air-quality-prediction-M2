# milestone3_dashboard.py

import streamlit as st
import pandas as pd
from aqi import compute_overall_aqi, get_bp_table, generate_alert

st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

st.title("Air Quality Prediction & Alerts (Milestone 3)")

# --- Upload pollutant CSV ---
uploaded_file = st.file_uploader("Upload CSV with pollutant forecast", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Optional: select row to compute AQI for
    row_idx = st.number_input("Select row index to compute AQI:", min_value=0, max_value=len(df)-1, value=0)

    row_data = df.iloc[row_idx]

    # Prepare pollutant dictionary
    pollutants = ["PM2.5","PM10","NO2","SO2","CO","O3"]
    data_dict = {poll: row_data[poll] if poll in row_data else None for poll in pollutants}

    # --- Compute AQI ---
    overall_aqi, sub_indices, category = compute_overall_aqi(data_dict)

    st.subheader("AQI Results")
    st.write(f"**Overall AQI:** {overall_aqi}")
    st.write(f"**Category:** {category}")
    st.write("**Pollutant Sub-Indices:**")
    st.json(sub_indices)

    # --- Alert Logic ---
    threshold = st.number_input("Set AQI Alert Threshold", min_value=0, max_value=500, value=200)
    alert = generate_alert(overall_aqi, threshold)
    if alert:
        st.warning(f"⚠️ ALERT! AQI {overall_aqi} exceeds threshold {threshold}")
    else:
        st.success(f"AQI {overall_aqi} is within safe limits")
