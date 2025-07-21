# 📁 model/lstm_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error

def run_lstm_forecast(df, forecast_days, currency="KSh"):
    df = df.copy()
    data = df[['Close']].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    sequence_length = 60
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)

    last_60_days = scaled_data[-60:]
    forecast = []
    for _ in range(forecast_days):
        input_data = last_60_days[-60:].reshape(1, 60, 1)
        predicted = model.predict(input_data, verbose=0)
        forecast.append(predicted[0][0])
        last_60_days = np.append(last_60_days, predicted)[-60:]

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    forecast_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)

    results = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
    if currency == "KSh":
        results['Forecast'] *= 157

    mae = mean_absolute_error(df['Close'][-forecast_days:], forecast[:len(df['Close'][-forecast_days:])])
    return results, mae
