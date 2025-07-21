import numpy as np

def calculate_backtest_metrics(pnl_series):
    pnl_series = pnl_series.dropna()
    daily_returns = pnl_series
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-9) * np.sqrt(252)
    sortino = np.mean(daily_returns) / (np.std(daily_returns[daily_returns < 0]) + 1e-9) * np.sqrt(252)
    cum_returns = (1 + daily_returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    return sharpe, sortino, max_dd
