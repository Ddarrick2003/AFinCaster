# 📁 model/xgboost_model.py

import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

def run_xgboost_with_shap(df, forecast_days, currency="KSh"):
    df = df.copy()

    # Ensure numeric values
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df.dropna(subset=['Close', 'Volume'], inplace=True)

    # Create target
    df['Target'] = df['Close'].shift(-forecast_days)
    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[features]
    y = df['Target']  # keep unscaled

    # Scale only X
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Train XGBoost
    model = xgb.XGBRegressor()
    model.fit(X_scaled, y)

    # Forecast input
    future_X = df[features].iloc[-forecast_days:]
    future_X_scaled = scaler_X.transform(future_X)
    predictions = model.predict(future_X_scaled)

    # Build forecast dataframe
    forecast_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': predictions})

    # If currency is KSh, apply exchange only if required (skip unless needed)
    # forecast_df['Forecast'] *= 157  # Comment out unless using USD data

    # MAE based on real prices
    actuals = df['Close'].iloc[-forecast_days:]
    mae = mean_absolute_error(actuals, forecast_df['Forecast'][:len(actuals)])

    # SHAP Explainability
    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, features=X, plot_type="bar", show=False)

    return forecast_df, mae, fig
