# 📁 model/transformer_models.py

import pandas as pd
import numpy as np

def run_informer(df, forecast_days, currency="KSh"):
    # Simulated placeholder until full model
    forecast_values = df['Close'].iloc[-1] + np.random.randn(forecast_days)
    forecast_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_values})
    if currency == "KSh":
        forecast_df['Forecast'] *= 157
    return forecast_df

def run_autoformer(df, forecast_days, currency="KSh"):
    # Simulated placeholder until full model
    forecast_values = df['Close'].iloc[-1] + np.cumsum(np.random.randn(forecast_days))
    forecast_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_values})
    if currency == "KSh":
        forecast_df['Forecast'] *= 157
    return forecast_df
