
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# --- Load Data ---
data_path = 'data/processed/cleaned_AQI_dataset.csv'
df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
print("Data loaded. Columns:", df.columns.tolist())

# --- Choose target pollutant ---
target = 'PM2.5'
series = df[target].dropna()
print(f"Training models for pollutant: {target}")

# --- ARIMA ---
try:
    arima_model = ARIMA(series, order=(2,1,2)).fit()
    arima_pred = arima_model.predict(start=0, end=len(series)-1, typ='levels')
    arima_rmse = np.sqrt(mean_squared_error(series, arima_pred))
    arima_mae = mean_absolute_error(series, arima_pred)
    print(f"ARIMA -> RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f}")
except Exception as e:
    print("ARIMA failed:", e)
    arima_model, arima_rmse, arima_mae = None, float('inf'), float('inf')

# --- Prophet ---
try:
    prophet_df = series.reset_index().rename(columns={'Date':'ds', target:'y'})
    prophet_model = Prophet()
    prophet_model.fit(prophet_df)
    forecast = prophet_model.predict(prophet_df)
    prophet_pred = forecast['yhat'].values
    prophet_rmse = np.sqrt(mean_squared_error(prophet_df['y'], prophet_pred))
    prophet_mae = mean_absolute_error(prophet_df['y'], prophet_pred)
    print(f"Prophet -> RMSE: {prophet_rmse:.2f}, MAE: {prophet_mae:.2f}")
except Exception as e:
    print("Prophet failed:", e)
    prophet_model, prophet_rmse, prophet_mae = None, float('inf'), float('inf')

# --- LSTM ---
try:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))
    X, y = [], []
    for i in range(1, len(scaled)):
        X.append(scaled[i-1])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    lstm_model = Sequential([
        LSTM(32, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X, y, epochs=20, batch_size=16, verbose=0, callbacks=[EarlyStopping(patience=3)])

    lstm_pred = lstm_model.predict(X).flatten()
    lstm_pred_inv = scaler.inverse_transform(lstm_pred.reshape(-1,1)).flatten()
    y_inv = scaler.inverse_transform(y.reshape(-1,1)).flatten()
    lstm_rmse = np.sqrt(mean_squared_error(y_inv, lstm_pred_inv))
    lstm_mae = mean_absolute_error(y_inv, lstm_pred_inv)
    print(f"LSTM -> RMSE: {lstm_rmse:.2f}, MAE: {lstm_mae:.2f}")
except Exception as e:
    print("LSTM failed:", e)
    lstm_model, scaler, lstm_rmse, lstm_mae = None, None, float('inf'), float('inf')

# --- Select best model ---
results = {'ARIMA': arima_rmse, 'Prophet': prophet_rmse, 'LSTM': lstm_rmse}
best_model_name = min(results, key=results.get)
print(f"Best model: {best_model_name}")

# --- Save best model ---
os.makedirs('models', exist_ok=True)
if best_model_name == 'ARIMA' and arima_model is not None:
    arima_model.save('models/best_model_arima.pkl')
elif best_model_name == 'Prophet' and prophet_model is not None:
    joblib.dump(prophet_model, 'models/best_model_prophet.pkl')
elif best_model_name == 'LSTM' and lstm_model is not None:
    lstm_model.save('models/best_model_lstm.h5')
    joblib.dump(scaler, 'models/lstm_scaler.pkl')

# --- Save evaluation report ---
with open('models/model_results.txt', 'w') as f:
    f.write(f"ARIMA -> RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f}\n")
    f.write(f"Prophet -> RMSE: {prophet_rmse:.2f}, MAE: {prophet_mae:.2f}\n")
    f.write(f"LSTM -> RMSE: {lstm_rmse:.2f}, MAE: {lstm_mae:.2f}\n")
    f.write(f"Best model: {best_model_name}\n")

print("Training complete. Models and results saved in /models/")
print("Script started!")

