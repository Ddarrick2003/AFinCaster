# 📁 model/xgboost_model.py

import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

def run_xgboost_with_shap(df, forecast_days, currency="KSh"):
    df = df.copy()
    
    # Ensure 'Volume' is numeric
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df.dropna(subset=['Volume'], inplace=True)
    
    # Create target column
    df['Target'] = df['Close'].shift(-forecast_days)
    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[features]
    y = df['Target']

    # Normalize
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Train XGBoost
    model = xgb.XGBRegressor()
    model.fit(X_scaled, y_scaled)

    # Forecast input
    future_X = df[features].iloc[-forecast_days:]
    future_X_scaled = scaler_X.transform(future_X)
    predictions_scaled = model.predict(future_X_scaled)

    # Inverse transform to get price in original scale
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    forecast_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': predictions})

    if currency == "KSh":
        forecast_df['Forecast'] *= 157  # Only apply if training data was in USD

    # Match last close for MAE
    actuals = df['Close'].iloc[-forecast_days:]
    mae = mean_absolute_error(actuals, forecast_df['Forecast'][:len(actuals)])

    # SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, features=X, plot_type="bar", show=False)
    
    return forecast_df, mae, fig
