import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from arch import arch_model

def run_garch_forecast(df, forecast_days, currency):
    df = df.copy()
    df = df.sort_values("Date")

    returns = 100 * df["Close"].pct_change().dropna()
    am = arch_model(returns, vol='GARCH', p=1, q=1)
    
    try:
        res = am.fit(disp='off')
        forecast = res.forecast(horizon=forecast_days)
        vol_forecast = np.sqrt(forecast.variance.values[-1, :])
        dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)

        st.subheader("📉 GARCH Volatility Forecast")
        st.line_chart(pd.Series(vol_forecast, index=dates, name="Volatility"))

        VaR_95 = -1.65 * vol_forecast
        st.subheader("📊 Value at Risk (95%)")
        st.line_chart(pd.Series(VaR_95, index=dates, name="VaR 95%"))

    except Exception as e:
        st.error(f"GARCH Error: {e}")
