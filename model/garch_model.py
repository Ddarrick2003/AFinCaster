import pandas as pd
from arch import arch_model

def run_garch_forecast(df, forecast_days, currency):
    df = df.copy()
    df['Return'] = df['Close'].pct_change().dropna()
    returns = df['Return'].dropna() * 100

    model = arch_model(returns, vol='Garch', p=1, q=1)
    garch_fit = model.fit(disp="off")

    forecast_result = garch_fit.forecast(horizon=forecast_days)
    volatility = forecast_result.variance.values[-1, :]
    predicted_returns = forecast_result.mean.values[-1, :]

    last_price = df['Close'].iloc[-1]
    forecast_prices = [last_price * (1 + predicted_returns[0]/100)]
    for i in range(1, forecast_days):
        next_price = forecast_prices[-1] * (1 + predicted_returns[i]/100)
        forecast_prices.append(next_price)

    forecast_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_prices})
    volatility_df = pd.DataFrame({'Date': forecast_dates, 'Volatility': volatility})

    return forecast_df, volatility_df
