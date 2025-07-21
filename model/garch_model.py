import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def run_xgboost_with_shap(df, forecast_days, currency):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna()

    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    features = ['Open', 'High', 'Low', 'Volume', 'Day', 'Month', 'Year']
    target = 'Close'

    X = df[features]
    y = df[target]
    
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X, y)

    future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    future_df = pd.DataFrame({
        'Open': [df['Open'].mean()] * forecast_days,
        'High': [df['High'].mean()] * forecast_days,
        'Low': [df['Low'].mean()] * forecast_days,
        'Volume': [df['Volume'].mean()] * forecast_days,
        'Day': future_dates.day,
        'Month': future_dates.month,
        'Year': future_dates.year,
    })

    future_preds = model.predict(future_df)
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_preds})

    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap_plot = shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Feature Importance")

    mae = mean_absolute_error(y, model.predict(X))

    return forecast_df, mae, plt
