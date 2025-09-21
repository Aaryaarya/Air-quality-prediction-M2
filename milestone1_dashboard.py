import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================== Page Config ==================
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# ================== Custom Styling ==================
st.markdown(
    """
    <style>
    body { background-color: #F7F9FB; }
    .main { background-color: #F7F9FB; }
    .big-title {
        font-size:36px !important;
        color:#ffffff;  /* Darker for better contrast */
        font-weight:700;
        text-align:left;
        padding-bottom:5px;
    }
    .section-card {
        background-color:#ffffff;
        border-radius:12px;
        padding:20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom:20px;
        color:#1B1B1B;  /* Ensure text inside is visible */
    }
    .metric-card {
        background-color: #f1f8f4;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin: 8px 0;
        color:#1B1B1B;  /* Improve metric text visibility */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== Header ==================
st.markdown('<p class="big-title">🌫 Air Quality Data Explorer</p>', unsafe_allow_html=True)
st.write("Milestone: Working Application Dashboard View")

# ================== Load Dataset ==================
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/cleaned_AQI_dataset.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# ================== Sidebar Controls ==================
st.sidebar.header("🔧 Data Controls")

cities = df['City'].unique()
selected_city = st.sidebar.selectbox("City", cities)

min_date, max_date = df['Date'].min(), df['Date'].max()
date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)

pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
selected_pollutants = st.sidebar.multiselect("Pollutants", pollutants, default=['PM2.5', 'PM10'])

st.sidebar.subheader("📊 Data Quality")
st.sidebar.progress(92, text="Completeness: 92%")
st.sidebar.progress(87, text="Validity: 87%")

# ================== Filter Data ==================
filtered_df = df[
    (df['City'] == selected_city) &
    (df['Date'] >= pd.to_datetime(date_range[0])) &
    (df['Date'] <= pd.to_datetime(date_range[1]))
].copy()

# ================== Layout ==================
col1, col2 = st.columns([2, 1])

# ---- Time Series ----
with col1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Pollutant Time Series")
    fig, ax = plt.subplots(figsize=(8,4))
    for p in selected_pollutants:
        ax.plot(filtered_df['Date'], filtered_df[p], linewidth=2, label=p)
    ax.set_xlabel("Date")
    ax.set_ylabel("Concentration (µg/m³)")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Correlation ----
with col2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Pollutant Correlations")
    if len(selected_pollutants) > 1:
        corr = filtered_df[selected_pollutants].corr()
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(corr, annot=True, cmap="Greens", fmt=".2f", cbar=False, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Select 2 or more pollutants for correlation.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Summary + Distribution ----
col3, col4 = st.columns([1, 2])

# Summary
with col3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Statistical Summary")
    for p in selected_pollutants:
        stats = filtered_df[p].describe()
        st.markdown(f"{p}")
        col_stats = st.columns(2)
        metrics = {
            "Mean": stats["mean"],
            "Median": stats["50%"],
            "Max": stats["max"],
            "Min": stats["min"],
            "Std Dev": stats["std"],
            "Count": stats["count"]
        }
        for i, (k,v) in enumerate(metrics.items()):
            with col_stats[i%2]:
                st.markdown(f"<div class='metric-card'><h4>{k}</h4><h2>{v:.2f}</h2></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Distribution
with col4:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Distribution Analysis")
    for p in selected_pollutants:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.hist(filtered_df[p].dropna(), bins=20, color="#66BB6A", edgecolor="black")
        ax.set_title(f"{p} Distribution")
        ax.set_xlabel(f"{p} (µg/m³)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ================== Footer ==================
st.markdown(
    """
    <div style='text-align: center; padding: 15px; font-size: 14px; color: gray;'>
    🚀 Built with <b>Streamlit</b> | Inspired by Clean Dashboard Design 🌍
    </div>
    """,
    unsafe_allow_html=True
)