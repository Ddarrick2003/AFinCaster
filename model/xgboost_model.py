import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import shap
import matplotlib.pyplot as plt

def run_xgboost_with_shap(df, forecast_days, currency):
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Target'] = df['Close'].shift(-forecast_days)
    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[features]
    y = df['Target']

    model = xgb.XGBRegressor()
    model.fit(X, y)

    prediction = model.predict(X.tail(1).values.reshape(1, -1))[0]
    forecast_df = pd.DataFrame({
        'Date': [df['Date'].iloc[-1] + pd.Timedelta(days=forecast_days)],
        'Predicted_Price': [prediction]
    })
    forecast_df['Price'] = forecast_df['Predicted_Price']
    forecast_df['Price'] = forecast_df['Price'].apply(lambda x: x * 144 if currency == "KSh" else x)

    mae = mean_absolute_error(y, model.predict(X))

    # SHAP plotting
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap_plot = shap.plots.bar(shap_values, show=False)
    fig = plt.gcf()

    return forecast_df, mae, fig
