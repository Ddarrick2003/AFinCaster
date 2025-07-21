# ========================
# 📁 FILE: model/garch_model.py
# ========================

import pandas as pd
import numpy as np
from arch import arch_model

def run_garch_forecast(df, forecast_days=10, currency="KSh"):
    df = df.copy()
    df = df[['Date', 'Close']].dropna()
    df.set_index('Date', inplace=True)

    returns = 100 * df['Close'].pct_change().dropna()
    am = arch_model(returns, vol='Garch', p=1, q=1)
    
    try:
        res = am.fit(disp="off")
    except Exception as e:
        raise RuntimeError(f"GARCH Error: {e}")

    forecast = res.forecast(horizon=forecast_days)
    variance_forecast = forecast.variance.values[-1, :]
    mean_forecast = forecast.mean.values[-1, :]

    last_price = df['Close'].iloc[-1]
    price_forecast = last_price * (1 + (mean_forecast / 100)).cumprod()

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': price_forecast})
    volatility_df = pd.DataFrame({'Date': future_dates, 'Volatility': np.sqrt(variance_forecast)})

    forecast_df['Forecast'] = forecast_df['Forecast'].astype(float)
    volatility_df['Volatility'] = volatility_df['Volatility'].astype(float)

    return forecast_df, volatility_df
