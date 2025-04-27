# Stock Market Analysis with Prediction

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# --- Config ---
ticker = 'TSLA'
start_date = '2023-01-01'
end_date = '2024-01-01'

# --- Load Data ---
stock_data = yf.download(ticker, start=start_date, end=end_date)
stock_data.dropna(inplace=True)

# --- Basic Info ---
print(stock_data.head())
print(stock_data.describe())

# --- Plot Closing Price ---
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='TSLA Close Price')
plt.title(f'{ticker} Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# --- Moving Averages ---
stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()

plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='Close Price', color='gray')
plt.plot(stock_data['MA20'], label='20-Day MA', color='green')
plt.plot(stock_data['MA50'], label='50-Day MA', color='orange')
plt.title(f'{ticker} - Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# --- Daily Return ---
stock_data['Daily Return'] = stock_data['Close'].pct_change()
plt.figure(figsize=(10, 5))
stock_data['Daily Return'].plot(kind='hist', bins=50, color='purple', edgecolor='black')
plt.title(f'{ticker} - Daily Return Distribution')
plt.xlabel('Daily Return')
plt.grid(True)
plt.show()

# --- Candlestick Chart ---
recent_data = stock_data[-60:]
recent_data.index.name = 'Date'
mpf.plot(recent_data, type='candle', style='yahoo', title=f"{ticker} Candlestick - Last 60 Days", volume=True, mav=(20, 50))

# --- Bollinger Bands ---
stock_data['Upper Band'] = stock_data['MA20'] + 2*stock_data['Close'].rolling(window=20).std()
stock_data['Lower Band'] = stock_data['MA20'] - 2*stock_data['Close'].rolling(window=20).std()

plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='Close Price')
plt.plot(stock_data['MA20'], label='20-Day MA')
plt.plot(stock_data['Upper Band'], label='Upper Band')
plt.plot(stock_data['Lower Band'], label='Lower Band')
plt.fill_between(stock_data.index, stock_data['Upper Band'], stock_data['Lower Band'], color='lightgray', alpha=0.3)
plt.title(f'{ticker} Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# --- RSI ---
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

stock_data['RSI'] = compute_rsi(stock_data['Close'])

plt.figure(figsize=(12, 4))
plt.plot(stock_data['RSI'], color='orange')
plt.axhline(70, linestyle='--', color='red')
plt.axhline(30, linestyle='--', color='green')
plt.title(f'{ticker} RSI')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.grid(True)
plt.show()

# --- Machine Learning Prediction ---
stock_data['Target'] = stock_data['Close'].shift(-1)
stock_data.dropna(inplace=True)

X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = stock_data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Price')
plt.plot(y_pred, label='Predicted Price')
plt.title(f'{ticker} Price Prediction (Linear Regression)')
plt.xlabel('Test Data Points')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
