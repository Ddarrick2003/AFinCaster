# 📁 model/xgboost_model.py

import pandas as pd
import xgboost as xgb
import shap
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def run_xgboost_with_shap(df, forecast_days, currency="KSh"):
    df = df.copy()

    # Clean 'Volume' column: Convert to numeric
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df.dropna(subset=['Volume'], inplace=True)

    # Create target column
    df['Target'] = df['Close'].shift(-forecast_days)
    df.dropna(inplace=True)

    # Features and labels
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[features]
    y = df['Target']

    # Train XGBoost model
    model = xgb.XGBRegressor()
    model.fit(X, y)

    # Forecast future prices using last known inputs
    future_X = df[features].iloc[-forecast_days:]
    predictions = model.predict(future_X)

    # Create forecast DataFrame
    forecast_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': predictions})

    # Compute MAE over last known forecast_days period
    actual = df['Close'].iloc[-forecast_days:]
    predicted = predictions[:len(actual)]
    mae = mean_absolute_error(actual, predicted)

    # Apply KSh conversion if requested
    if currency == "KSh":
        forecast_df['Forecast'] = forecast_df['Forecast'] * 157
        predictions = predictions * 157  # Also convert predictions for display

    # SHAP explainability
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()

    return forecast_df, mae, fig, predictions[0]  # Return next-day predicted value
