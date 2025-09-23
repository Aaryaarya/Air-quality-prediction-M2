
import os
import pandas as pd
import numpy as np
import joblib

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Create folder to save best models
os.makedirs('bestmodel', exist_ok=True)

# Load dataset
df = pd.read_csv('data/processed/cleaned_AQI_dataset.csv', parse_dates=['Date'], index_col='Date')

pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
all_metrics = {}

def evaluate(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Handle divide by zero for MAPE
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mape = np.nan_to_num(mape, nan=np.inf)
    smape = 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'smape': smape}

# --- MODEL FUNCTIONS ---
def train_arima(series):
    model = ARIMA(series, order=(2,1,2))
    res = model.fit()
    pred = res.predict(start=0, end=len(series)-1, typ='levels')
    metrics = evaluate(series, pred)
    return model, metrics

def train_prophet(series):
    df_prophet = series.reset_index().rename(columns={'Date':'ds', series.name:'y'})
    model = Prophet()
    model.fit(df_prophet)
    forecast = model.predict(df_prophet)
    pred = forecast['yhat'].values
    metrics = evaluate(df_prophet['y'].values, pred)
    return model, metrics

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
    return model, scaler, metrics

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
    return model, metrics

# --- LOOP THROUGH POLLUTANTS ---
for pollutant in pollutants:
    print(f"\n--- Training models for {pollutant} ---")
    series = df[pollutant].dropna()
    if len(series) < 10:
        print(f"Not enough data for {pollutant}. Skipping...")
        continue
    
    # Train models
    arima_model, arima_metrics = train_arima(series)
    prophet_model, prophet_metrics = train_prophet(series)
    lstm_model, lstm_scaler, lstm_metrics = train_lstm(series)
    xgb_model, xgb_metrics = train_xgboost(series)
    
    # Select best model by RMSE
    model_metrics = {
        'ARIMA': arima_metrics['rmse'],
        'Prophet': prophet_metrics['rmse'],
        'LSTM': lstm_metrics['rmse'],
        'XGBoost': xgb_metrics['rmse']
    }
    best_model_name = min(model_metrics, key=model_metrics.get)
    print(f"Best model for {pollutant}: {best_model_name}")
    
    # Save best model
    if best_model_name == 'ARIMA':
        joblib.dump(arima_model, f'bestmodel/best_{pollutant}_ARIMA.pkl')
    elif best_model_name == 'Prophet':
        joblib.dump(prophet_model, f'bestmodel/best_{pollutant}_Prophet.pkl')
    elif best_model_name == 'LSTM':
        lstm_model.save(f'bestmodel/best_{pollutant}_LSTM.h5')
        joblib.dump(lstm_scaler, f'bestmodel/{pollutant}_LSTM_scaler.pkl')
    elif best_model_name == 'XGBoost':
        joblib.dump(xgb_model, f'bestmodel/best_{pollutant}_XGBoost.pkl')
    
    # Store all metrics
    all_metrics[pollutant] = {
        'ARIMA': arima_metrics,
        'Prophet': prophet_metrics,
        'LSTM': lstm_metrics,
        'XGBoost': xgb_metrics,
        'BestModel': best_model_name
    }

# Save all metrics as CSV for dashboard
metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
metrics_df.to_csv('bestmodel/all_metrics_summary.csv')
print("\n✅ All models and metrics saved in 'bestmodel/' folder")
