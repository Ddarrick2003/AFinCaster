# 📁 model/xgboost_model.py

import pandas as pd
import xgboost as xgb
import shap
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def run_xgboost_with_shap(df, forecast_days, currency="KSh"):
    df = df.copy()
    df['Target'] = df['Close'].shift(-forecast_days)
    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[features]
    y = df['Target']

    model = xgb.XGBRegressor()
    model.fit(X, y)

    future_X = df[features].iloc[-forecast_days:]
    predictions = model.predict(future_X)

    forecast_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': predictions})

    if currency == "KSh":
        forecast_df['Forecast'] *= 157

    mae = mean_absolute_error(df['Close'].iloc[-forecast_days:], predictions[:len(df['Close'].iloc[-forecast_days:])])

    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    return forecast_df, mae, fig
