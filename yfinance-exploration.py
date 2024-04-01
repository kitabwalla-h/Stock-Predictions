import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# get historical 1 year data for MSFT
stock_symbol = 'MSFT'
start_date = '2022-01-01'
end_date = '2023-01-01'
data = yf.download(stock_symbol, start=start_date, end=end_date)

# scale closing data on a level from 0 to 1
close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices)

# create lstm function - x is in the input and y is the output
def create_lstm_data(data, time_steps=1):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(x), np.array(y)

# set number of time steps - occurrences of taking data into consideration
time_steps = 10
x, y = create_lstm_data(close_prices_scaled, time_steps)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# model initializes a sequential neural network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Trains the LSTM model using the prepared input data (x) and target values (y).
model.fit(x, y, epochs=50, batch_size=32)

# generate future dates for prediction
future_dates = pd.date_range(start=end_date, periods=30)

# Extracts the last 10 closing prices to predict future prices.
last_prices = close_prices[-time_steps:]

# Scales the last prices and prepares the input data for prediction.
last_prices_scaled = scaler.transform(last_prices.reshape(-1, 1))

# Uses the trained LSTM model to predict future prices and inverse transforms the scaled predictions.
predicted_prices = []

for _ in range(len(future_dates)):
    x_pred = np.array([last_prices_scaled[-time_steps:, 0]])
    x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))
    predicted_price_scaled = model.predict(x_pred)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    predicted_prices.append(predicted_price[0][0])

# Create DataFrame for predicted prices with future dates
future_data = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices})

print(future_data)
