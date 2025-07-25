import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

def run_garch_forecast(df, forecast_days, currency="KSh"):
    df = df.copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)

    returns = 100 * df['Close'].pct_change().dropna()

    model = arch_model(returns, vol='Garch', p=1, q=1)
    res = model.fit(disp="off")
    forecasts = res.forecast(horizon=forecast_days)

    vol = forecasts.variance.values[-1, :]
    forecast_volatility = np.sqrt(vol)

    forecast_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': df['Close'].iloc[-1]})  # GARCH does not predict price
    volatility_df = pd.DataFrame({'Date': forecast_dates, 'Volatility': forecast_volatility})

    # Plot
    fig, ax = plt.subplots()
    ax.plot(volatility_df['Date'], volatility_df['Volatility'], marker='o', color='orange')
    ax.set_title("GARCH Forecasted Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.grid(True)

    return forecast_df, volatility_df
