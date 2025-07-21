import pandas as pd
import numpy as np

def run_informer(df, forecast_days, currency):
    last_price = df['Close'].iloc[-1]
    forecast = [last_price * (1 + np.random.normal(0, 0.01)) for _ in range(forecast_days)]
    forecast_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    return pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})

def run_autoformer(df, forecast_days, currency):
    last_price = df['Close'].iloc[-1]
    forecast = [last_price * (1 + np.random.normal(0, 0.015)) for _ in range(forecast_days)]
    forecast_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    return pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
