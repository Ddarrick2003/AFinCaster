# utils/preprocessing.py

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD

def preprocess_data(df):
    df = df.copy()

    # Remove commas in price columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False).astype(float)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # Drop bad date rows

    df['Log_Volume'] = np.log1p(df['Volume'])

    result = []

    for ticker in df['Ticker'].unique():
        dft = df[df['Ticker'] == ticker].sort_values('Date').copy()

        # Calculate technical indicators
        try:
            dft['RSI'] = RSIIndicator(close=dft['Close'], window=14).rsi()
            macd = MACD(close=dft['Close'])
            dft['MACD'] = macd.macd()
            dft['MACD_Signal'] = macd.macd_signal()
        except Exception as e:
            dft['RSI'] = dft['MACD'] = dft['MACD_Signal'] = np.nan

        dft['Returns'] = dft['Close'].pct_change().fillna(0)

        result.append(dft)

    df_final = pd.concat(result)
    df_final = df_final.dropna(subset=['RSI', 'MACD', 'MACD_Signal'])

    return df_final.reset_index(drop=True)
