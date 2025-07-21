import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

def run_lstm_forecast(df, forecast_days, currency):
    df = df.copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)

    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    sequence_length = 60
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])
    x, y = np.array(x), np.array(y)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, y, epochs=10, batch_size=32, verbose=0)

    input_data = scaled_data[-sequence_length:]
    forecast = []
    input_seq = input_data.reshape(1, sequence_length, 1)

    for _ in range(forecast_days):
        next_pred = model.predict(input_seq, verbose=0)
        forecast.append(next_pred[0, 0])
        input_seq = np.append(input_seq[:, 1:, :], [[next_pred[0]]], axis=1)

    forecast_prices = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_days)]

    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_prices})
    mae = mean_absolute_error(df['Close'][-len(y):], scaler.inverse_transform(y))

    return forecast_df, mae
