import pandas as pd
from arch import arch_model

def run_garch_forecast(df, forecast_days, currency):
    df = df[['Date', 'Close']].copy()
    df.set_index('Date', inplace=True)
    returns = 100 * df['Close'].pct_change().dropna()

    model = arch_model(returns, vol='Garch', p=1, q=1)
    res = model.fit(disp="off")
    forecast = res.forecast(horizon=forecast_days)

    volatility = forecast.variance.values[-1, :] ** 0.5
    last_price = df['Close'].iloc[-1]
    forecast_prices = [last_price * (1 + 0.01 * vol) for vol in volatility]

    forecast_dates = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='D')[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_Price': forecast_prices})
    forecast_df['Price'] = forecast_df['Predicted_Price']
    forecast_df['Price'] = forecast_df['Price'].apply(lambda x: x * 144 if currency == "KSh" else x)

    return forecast_df, volatility
