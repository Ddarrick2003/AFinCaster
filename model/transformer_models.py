import pandas as pd
import numpy as np

def run_informer(df, forecast_days, currency):
    # Placeholder for Informer logic
    last_price = df['Close'].iloc[-1]
    forecast_prices = np.linspace(last_price, last_price * 1.02, forecast_days)

    forecast_dates = pd.date_range(df['Date'].iloc[-1], periods=forecast_days + 1, freq='D')[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_Price': forecast_prices})
    forecast_df['Price'] = forecast_df['Predicted_Price']
    forecast_df['Price'] = forecast_df['Price'].apply(lambda x: x * 144 if currency == "KSh" else x)

    return forecast_df


def run_autoformer(df, forecast_days, currency):
    # Placeholder for Autoformer logic
    last_price = df['Close'].iloc[-1]
    forecast_prices = np.linspace(last_price, last_price * 1.03, forecast_days)

    forecast_dates = pd.date_range(df['Date'].iloc[-1], periods=forecast_days + 1, freq='D')[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_Price': forecast_prices})
    forecast_df['Price'] = forecast_df['Predicted_Price']
    forecast_df['Price'] = forecast_df['Price'].apply(lambda x: x * 144 if currency == "KSh" else x)

    return forecast_df
