import streamlit as st
import pandas as pd
import plotly.express as px
from aqi import compute_overall_aqi

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
st.title("🌍 Air Quality Alert System (Milestone 3)")

# --- CSV Upload ---
uploaded_file = st.file_uploader("Upload CSV with pollutant forecast", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # --- Compute AQI for all rows ---
    pollutants = ["PM2.5","PM10","NO2","SO2","CO","O3"]
    aqi_list = []
    category_list = []

    for _, row in df.iterrows():
        data_dict = {poll: row[poll] if poll in row else None for poll in pollutants}
        overall_aqi, _, category = compute_overall_aqi(data_dict)
        aqi_list.append(overall_aqi)
        category_list.append(category)

    df['AQI'] = aqi_list
    df['Category'] = category_list

    st.subheader("Computed AQI & Category")
    st.dataframe(df[['AQI','Category']].head())

    # --- Alert Threshold ---
    threshold = st.number_input("Set AQI Alert Threshold", min_value=0, max_value=500, value=200)
    alert_rows = df[df['AQI'] > threshold]

    if not alert_rows.empty:
        st.warning(f"⚠️ {len(alert_rows)} rows exceed AQI threshold {threshold}")
    else:
        st.success(f"All AQI values are below threshold {threshold}")

    # --- Pollutant Filter ---
    st.subheader("📊 Select Pollutants (Optional for Hover Info)")
    selected_pollutants = st.multiselect(
        "Select pollutants to display in hover:",
        options=pollutants,
        default=pollutants
    )

    # --- AQI Trend Graph ---
    st.subheader("📈 AQI Trend with Categories and Alerts")
    df['Color'] = df['Category'].map({
        "Good":"green","Satisfactory":"lightgreen","Moderate":"yellow",
        "Poor":"orange","Very Poor":"red","Severe":"purple","Unknown":"grey"
    })

    fig = px.scatter(
        df,
        x=df.index,
        y='AQI',
        color='Category',
        color_discrete_map={
            "Good":"green","Satisfactory":"lightgreen","Moderate":"yellow",
            "Poor":"orange","Very Poor":"red","Severe":"purple","Unknown":"grey"
        },
        hover_data=selected_pollutants
    )

    # Highlight alerts on AQI trend
    if not alert_rows.empty:
        fig.add_scatter(
            x=alert_rows.index,
            y=alert_rows['AQI'],
            mode='markers',
            marker=dict(color='red', size=12, symbol='x'),
            name='Alert'
        )

    fig.update_layout(
        xaxis_title="Row Index / Time",
        yaxis_title="AQI",
        legend_title="Category",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Active Alerts Section ---
    st.subheader("⚠️ Active Alerts")
    if not alert_rows.empty:
        st.dataframe(alert_rows[['AQI','Category']].rename(columns={"AQI":"AQI Value"}))

        # --- Download Alerts CSV ---
        csv = alert_rows[['AQI','Category']].to_csv(index_label="Row Index")
        st.download_button(
            label="Download Active Alerts CSV",
            data=csv,
            file_name="active_alerts.csv",
            mime="text/csv"
        )
    else:
        st.success("No active alerts.")
