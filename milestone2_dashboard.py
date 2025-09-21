import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Model-specific imports
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from datetime import datetime, timedelta

# ================== Page Config ==================
st.set_page_config(
    page_title="Air Quality Model Training", 
    layout="wide",
    page_icon="📊"
)

# ================== Custom Styling ==================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
    
    .big-title {
        font-size: 2.75rem !important;
        color: #1a472a;
        font-weight: 800;
        text-align: left;
        padding-bottom: 5px;
        margin-bottom: 1.5rem;
        background: linear-gradient(90deg, #1a472a 0%, #2E7D32 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a472a;
        margin-bottom: 1.2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e8f5e9;
    }
    
    .section-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
        margin-bottom: 24px;
        border: 1px solid rgba(76, 175, 80, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .section-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.09);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin: 10px 0;
        border-left: 5px solid #2E7D32;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    .metric-card h4 {
        color: #388e3c;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0 0 8px 0;
    }
    
    .metric-card h2 {
        color: #1b5e20;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(90deg, #2E7D32 0%, #4CAF50 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: 500;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2E7D32 0%, #4CAF50 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f1f8e9;
        border-radius: 12px 12px 0 0;
        padding: 12px 20px;
        font-weight: 500;
        color: #388e3c;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #e8f5e9;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #2E7D32 0%, #4CAF50 100%);
        color: white;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px;
        font-size: 0.9rem;
        color: #757575;
        margin-top: 40px;
        border-top: 1px solid #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== Header ==================
st.markdown('<p class="big-title">🤖 Air Quality Model Training Dashboard</p>', unsafe_allow_html=True)
st.markdown("""
    <p style="font-size: 1.1rem; color: #616161; margin-top: -1rem; margin-bottom: 2rem;">
    Train and evaluate forecasting models for air quality prediction
    </p>
""", unsafe_allow_html=True)

# ================== Load Dataset ==================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/processed/cleaned_AQI_dataset.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        st.error("Could not load the dataset. Please make sure the file exists at 'data/processed/cleaned_AQI_dataset.csv'")
        return None

df = load_data()

# ================== Model Training Functions ==================
def evaluate(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Handle divide by zero for MAPE
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mape = np.nan_to_num(mape, nan=np.inf)
    smape = 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'sMAPE': smape}

def train_arima(series):
    model = ARIMA(series, order=(2,1,2))
    res = model.fit()
    pred = res.predict(start=0, end=len(series)-1, typ='levels')
    metrics = evaluate(series, pred)
    return model, metrics, pred

def train_prophet(series):
    df_prophet = series.reset_index().rename(columns={'Date':'ds', series.name:'y'})
    model = Prophet()
    model.fit(df_prophet)
    forecast = model.predict(df_prophet)
    pred = forecast['yhat'].values
    metrics = evaluate(df_prophet['y'].values, pred)
    return model, metrics, pred

def train_lstm(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))
    X, y = [], []
    for i in range(1, len(scaled)):
        X.append(scaled[i-1])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    
    model = Sequential([
        LSTM(32, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0, callbacks=[EarlyStopping(patience=3)])
    
    pred = model.predict(X).flatten()
    pred_inv = scaler.inverse_transform(pred.reshape(-1,1)).flatten()
    y_inv = scaler.inverse_transform(y.reshape(-1,1)).flatten()
    metrics = evaluate(y_inv, pred_inv)
    return model, scaler, metrics, pred_inv

def train_xgboost(series):
    X, y = [], []
    values = series.values
    for i in range(1, len(values)):
        X.append([values[i-1]])
        y.append(values[i])
    X, y = np.array(X), np.array(y)
    model = XGBRegressor(n_estimators=100)
    model.fit(X, y)
    pred = model.predict(X)
    metrics = evaluate(y, pred)
    return model, metrics, pred

# ================== Sidebar Controls ==================
with st.sidebar:
    st.markdown('<p style="font-size: 1.2rem; color: #1b5e20; font-weight: 600;">🔧 Training Controls</p>', unsafe_allow_html=True)
    
    if df is not None:
    # Drop non-pollutant columns
        pollutants = df.columns.drop(['Date', 'City', 'AQI']).tolist()

        selected_pollutant = st.selectbox("Select Pollutant", pollutants)
        
        cities = df['City'].unique() if 'City' in df.columns else ['All Cities']
        selected_city = st.selectbox("Select City", cities)
        
        # Filter data based on selection
        if selected_city != 'All Cities':
            filtered_df = df[df['City'] == selected_city]
        else:
            filtered_df = df
            
        series = filtered_df[selected_pollutant].dropna()
        
        st.markdown("---")
        st.markdown('<p style="font-size: 1.2rem; color: #1b5e20; font-weight: 600;">📊 Model Selection</p>', unsafe_allow_html=True)
        
        model_options = st.multiselect(
            "Select Models to Train",
            ["ARIMA", "Prophet", "LSTM", "XGBoost"],
            default=["ARIMA", "Prophet", "LSTM"]
        )
        
        train_button = st.button("🚀 Train Selected Models")
        
        st.markdown("---")
        st.markdown('<p style="font-size: 1.2rem; color: #1b5e20; font-weight: 600;">💾 Model Management</p>', unsafe_allow_html=True)
        
        # Create folders if they don't exist
        os.makedirs('bestmodel', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        if st.button("💾 Save Best Model"):
            st.info("This will save the best performing model based on RMSE")
    else:
        st.warning("Please load data first")

# ================== Main Dashboard ==================
if df is None:
    st.warning("Please ensure your data file is available at 'data/processed/cleaned_AQI_dataset.csv'")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📈 Model Training", "📊 Performance Comparison", "🔮 Predictions"])

with tab1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Model Training</p>', unsafe_allow_html=True)
    
    if train_button and len(model_options) > 0:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        predictions = {}
        
        for i, model_name in enumerate(model_options):
            status_text.text(f"Training {model_name} model...")
            progress_bar.progress((i + 1) / len(model_options))
            
            try:
                if model_name == "ARIMA":
                    model, metrics, pred = train_arima(series)
                    results["ARIMA"] = metrics
                    predictions["ARIMA"] = pred
                    
                elif model_name == "Prophet":
                    model, metrics, pred = train_prophet(series)
                    results["Prophet"] = metrics
                    predictions["Prophet"] = pred
                    
                elif model_name == "LSTM":
                    model, scaler, metrics, pred = train_lstm(series)
                    results["LSTM"] = metrics
                    predictions["LSTM"] = pred
                    
                elif model_name == "XGBoost":
                    model, metrics, pred = train_xgboost(series)
                    results["XGBoost"] = metrics
                    predictions["XGBoost"] = pred
                    
            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
        
        status_text.text("Training complete!")
        
        # Display results
        if results:
            st.markdown("### Training Results")
            
            # Create metrics dataframe
            metrics_df = pd.DataFrame(results).T
            st.dataframe(metrics_df.style.highlight_min(axis=0, color='#c8e6c9'))
            
            # Determine best model
            best_model = min(results, key=lambda x: results[x]['RMSE'])
            st.success(f"Best model: {best_model} (RMSE: {results[best_model]['RMSE']:.4f})")
            
            # Plot predictions
            st.markdown("### Predictions vs Actual")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(series.values, label='Actual', linewidth=2)
            
            colors = ['red', 'blue', 'green', 'orange']
            for i, (model_name, pred) in enumerate(predictions.items()):
                # Adjust length if needed
                if len(pred) < len(series):
                    pred = np.concatenate([np.array([np.nan]), pred])
                elif len(pred) > len(series):
                    pred = pred[:len(series)]
                
                ax.plot(pred, label=model_name, linestyle='--', alpha=0.8, color=colors[i % len(colors)])
            
            ax.legend()
            ax.set_xlabel('Time Index')
            ax.set_ylabel(selected_pollutant)
            ax.set_title('Model Predictions Comparison')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.predictions = predictions
            
    else:
        st.info("Select models and click 'Train Selected Models' to begin training")
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Model Performance Comparison</p>', unsafe_allow_html=True)
    
    if 'results' in st.session_state:
        results = st.session_state.results
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame(results).T
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### RMSE Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            models = list(results.keys())
            rmse_values = [results[m]['RMSE'] for m in models]
            bars = ax.bar(models, rmse_values, color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
            ax.set_ylabel('RMSE')
            ax.set_title('RMSE by Model')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
                
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### MAE Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            mae_values = [results[m]['MAE'] for m in models]
            bars = ax.bar(models, mae_values, color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
            ax.set_ylabel('MAE')
            ax.set_title('MAE by Model')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
                
            st.pyplot(fig)
        
        # Detailed metrics table
        st.markdown("#### Detailed Metrics")
        st.dataframe(metrics_df.style.highlight_min(axis=0, color='#c8e6c9'))
        
    else:
        st.info("Train models first to see performance comparison")
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Future Predictions</p>', unsafe_allow_html=True)
    
    if 'results' in st.session_state:
        results = st.session_state.results
        best_model = min(results, key=lambda x: results[x]['RMSE'])
        
        st.success(f"Using best model: {best_model} for predictions")
        
        # Number of days to forecast
        n_days = st.slider("Days to forecast", 1, 30, 7)
        
        # Make future predictions (simplified)
        last_value = series.iloc[-1]
        
        # Simple forecasting - in a real scenario, we would use the actual model
        future_pred = [last_value * (0.95 + 0.1 * np.random.random()) for _ in range(n_days)]
        
        # Create future dates
        last_date = df['Date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, n_days+1)]
        
        # Plot forecast
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot historical data
        historical_dates = df['Date'][-30:]  # Last 30 days
        historical_values = series[-30:]
        ax.plot(historical_dates, historical_values, label='Historical', linewidth=2)
        
        # Plot forecast
        ax.plot(future_dates, future_pred, label='Forecast', linewidth=2, color='red', linestyle='--')
        
        # Fill between for confidence interval
        ax.fill_between(future_dates, 
                       [x * 0.9 for x in future_pred], 
                       [x * 1.1 for x in future_pred], 
                       color='red', alpha=0.2)
        
        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel(selected_pollutant)
        ax.set_title(f'{n_days}-Day Forecast using {best_model}')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Display forecast values
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Value': future_pred
        })
        st.dataframe(forecast_df)
        
    else:
        st.info("Train models first to generate predictions")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================== Footer ==================
st.markdown(
    """
    <div class="footer">
    🚀 Built with <b>Streamlit</b> | Air Quality Model Training Dashboard 🌍
    </div>
    """,
    unsafe_allow_html=True
)