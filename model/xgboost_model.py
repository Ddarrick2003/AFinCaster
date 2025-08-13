import pandas as pd
import xgboost as xgb
import shap
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

def run_xgboost_with_shap(df, forecast_days, currency="KSh"):
    df = df.copy()

    # Ensure proper types
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df.dropna(subset=['Close', 'Volume'], inplace=True)

    # Adjust forecast_days if dataset is small
    if len(df) <= forecast_days:
        forecast_days = len(df) - 1

    # Create target
    df['Target'] = df['Close'].shift(-forecast_days)
    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[features]
    y = df['Target']

    # Scale features
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Train XGBoost
    model = xgb.XGBRegressor()
    model.fit(X_scaled, y)

    # Forecast
    future_X = df[features].iloc[-forecast_days:]
    future_X_scaled = scaler_X.transform(future_X)
    predictions = model.predict(future_X_scaled)

    forecast_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': predictions})
    forecast_df['Forecast'] = pd.to_numeric(forecast_df['Forecast'], errors='coerce')

    # Compute MAE
    actuals = df['Close'].iloc[-forecast_days:]
    mae = mean_absolute_error(actuals, forecast_df['Forecast'][:len(actuals)])

    # SHAP Explainability
    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)

    # Interactive SHAP bar chart
    shap_mean = pd.DataFrame({
        'Feature': features,
        'SHAP Value': abs(shap_values.values).mean(axis=0)
    }).sort_values(by='SHAP Value', ascending=True)

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        x=shap_mean['SHAP Value'],
        y=shap_mean['Feature'],
        orientation='h',
        marker=dict(color=shap_mean['SHAP Value'], colorscale='Viridis'),
        text=[f"{val:.4f}" for val in shap_mean['SHAP Value']],
        textposition='auto',
        hovertemplate='Feature: %{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>'
    ))
    fig_shap.update_layout(
        title="🔹 XGBoost Feature Importance (Mean |SHAP|)",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="Feature",
        template="plotly_white",
        height=400
    )

    # Predicted vs Actual chart
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=df['Date'].iloc[-forecast_days:],
        y=actuals,
        mode='lines+markers',
        name="Actual Close",
        line=dict(color='green')
    ))
    fig_pred.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Forecast'],
        mode='lines+markers',
        name="XGBoost Forecast",
        line=dict(color='blue', dash='dash')
    ))
    fig_pred.update_layout(
        title=f"📈 XGBoost Forecast vs Actual (MAE: {mae:.4f} {currency})",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})",
        template="plotly_white",
        height=400
    )

    # Return 4 values
    return forecast_df, mae, fig_shap, fig_pred
