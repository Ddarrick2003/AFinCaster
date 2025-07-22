# 📁 model/xgboost_model.py

import pandas as pd
import xgboost as xgb
import shap
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def run_xgboost_with_shap(df, forecast_days, currency="KSh"):
    df = df.copy()

    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    # Drop rows with missing values
    df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

    # Target is Close price shifted by forecast_days
    df['Target'] = df['Close'].shift(-forecast_days)
    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[features]
    y = df['Target']

    # Train XGBoost
    model = xgb.XGBRegressor()
    model.fit(X, y)

    # Predict future values using the last available rows
    future_X = df[features].iloc[-forecast_days:]
    future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    predictions = model.predict(future_X)

    # Currency conversion if required
    if currency == "KSh":
        predictions = predictions * 157

    # Prepare forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': predictions
    })

    # Calculate MAE (aligning prediction with last actuals)
    try:
        mae_window = min(len(df), forecast_days)
        actuals = df['Close'].iloc[-mae_window:]
        preds_for_mae = model.predict(X.iloc[-mae_window:])
        if currency == "KSh":
            preds_for_mae = preds_for_mae * 157
        mae = mean_absolute_error(actuals, preds_for_mae)
    except Exception:
        mae = None

    # SHAP Feature Importance
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    return forecast_df, mae, fig
