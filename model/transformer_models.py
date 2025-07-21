import pandas as pd
import numpy as np
from datetime import timedelta

def run_transformer_model(df, model_type="informer", forecast_horizon=10):
    # Placeholder logic for demo — replace with real Transformer model inference
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Close']].dropna()

    actual = df['Close']
    last_value = actual.iloc[-1]

    # Fake prediction: linear increase
    predicted = [last_value * (1 + 0.01 * i) for i in range(1, forecast_horizon + 1)]
    forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_horizon)

    return actual, predicted, forecast_dates
