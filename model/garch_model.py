# 📁 model/garch_model.py

import pandas as pd
import numpy as np
from arch import arch_model

def run_garch_forecast(df, forecast_days, currency="KSh"):
    df = df.copy()
    df['Returns'] = df['Close'].pct_change().dropna()
    returns = df['Returns'].dropna() * 100

    model = arch_model(returns, vol='Garch', p=1, q=1)
    res = model.fit(disp='off')
    forecast = res.forecast(horizon=forecast_days)

    forecast_variance = forecast.variance.values[-1, :]
    forecast_volatility = np.sqrt(forecast_variance)

    last_price = df['Close'].iloc[-1]
    simulated_prices = [last_price * (1 + np.random.normal(0, vol / 100)) for vol in forecast_volatility]

    forecast_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': simulated_prices, 'Volatility': forecast_volatility})

    if currency == "KSh":
        forecast_df['Forecast'] *= 157

    return forecast_df[['Date', 'Forecast']], forecast_df['Volatility']
