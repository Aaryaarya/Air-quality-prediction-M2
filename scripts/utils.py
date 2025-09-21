# scripts/utils.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_cleaned(path):
    """Load cleaned merged file; expects Date, City, pollutant columns."""
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def series_for_city_pollutant(df, city, pollutant, freq='D'):
    """Return a pandas Series indexed by Date, interpolated/forward-filled.
       freq='D' makes it daily; adjust if your data has different freq."""
    dfc = df[df['City'] == city][['Date', pollutant]].dropna().copy()
    if dfc.empty:
        return pd.Series(dtype=float)
    s = dfc.set_index('Date').sort_index()[pollutant]
    # reindex to regular frequency (daily) to avoid mismatch during forecasting
    s = s.asfreq(freq)
    s = s.interpolate(limit_direction='both').ffill().bfill()
    s.name = pollutant
    return s

def expanding_window_splits(n, initial_train, horizon, step):
    """
    Yield (train_slice, test_slice) as index ranges for expanding-window CV.
    - n: length of series
    - initial_train: number of points in first training window
    - horizon: forecast horizon (number of steps to predict each fold)
    - step: how many points to expand for each fold
    """
    start = initial_train
    while start + horizon <= n:
        train_idx = (0, start)          # python slice semantics: [0:start]
        test_idx = (start, start+horizon)
        yield train_idx, test_idx
        start += step

def evaluate(y_true, y_pred):
    """Return MAE, RMSE, MAPE, sMAPE."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE: avoid divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, np.nan, y_true))) * 100
    # sMAPE
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        smape = np.mean(np.where(denom==0, 0, np.abs(y_true - y_pred) / denom)) * 100
    return {'mae': float(mae), 'rmse': float(rmse), 'mape': float(np.nan_to_num(mape)), 'smape': float(np.nan_to_num(smape))}
