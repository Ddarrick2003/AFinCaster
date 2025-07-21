import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def run_lstm_forecast(df, forecast_days, currency):
    df = df[['Date', 'Close']].copy()
    df.set_index('Date', inplace=True)
    data = df.values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    look_back = 30
    X, y = [], []
    for i in range(look_back, len(scaled_data) - forecast_days):
        X.append(scaled_data[i - look_back:i])
        y.append(scaled_data[i:i + forecast_days])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(forecast_days))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y[:, :, 0], epochs=10, batch_size=16, verbose=0)

    recent_data = scaled_data[-look_back:]
    recent_data = np.expand_dims(recent_data, axis=0)
    prediction_scaled = model.predict(recent_data)[0]
    prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()

    forecast_dates = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='D')[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_Price': prediction})
    forecast_df['Price'] = forecast_df['Predicted_Price']
    forecast_df['Price'] = forecast_df['Price'].apply(lambda x: x * 144 if currency == "KSh" else x)

    actual = df['Close'].values[-forecast_days:]
    mae = np.mean(np.abs(prediction[:len(actual)] - actual))

    return forecast_df, mae
