import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import shap

def run_xgboost_forecast(df, forecast_days=10):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    df['Target'] = df['Close'].shift(-forecast_days)

    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # Predict future values using the last available inputs
    last_known = df.iloc[-forecast_days:][features]
    future_preds = model.predict(last_known)

    mae = mean_absolute_error(y_test, preds)

    shap_explainer = shap.Explainer(model, X_train)
    shap_values = shap_explainer(X_test)

    # Build comparison dataframe
    actuals = y_test.reset_index(drop=True)
    predicted = pd.Series(preds, name="Predicted")

    comparison_df = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predicted
    })

    return future_preds, comparison_df, mae, model, shap_values

