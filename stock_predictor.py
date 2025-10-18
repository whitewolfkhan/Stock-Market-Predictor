import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

def fetch_stock_data(ticker, start_date, end_date):
    """Fetches stock data from Yahoo Finance."""
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def preprocess_data(stock_data):
    """Preprocesses the stock data and adds technical indicators."""
    stock_data.dropna(inplace=True)
    
    # Moving Average
    stock_data['Moving_Average_50'] = stock_data['Close'].rolling(window=50).mean()
    
    # RSI
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = stock_data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = exp1 - exp2

    stock_data.dropna(inplace=True)
    return stock_data

def train_or_load_model(stock_data, ticker):
    """Trains a RandomForestRegressor model or loads a pre-trained one."""
    features = ['Open', 'High', 'Low', 'Volume', 'Moving_Average_50', 'RSI', 'MACD']
    target = 'Close'

    X = stock_data[features]
    y = stock_data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model_filename = f'{ticker}_model.joblib'

    if os.path.exists(model_filename):
        print("Loading existing model...")
        model = joblib.load(model_filename)
    else:
        print("Training new model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_filename)
        print(f"Model saved as {model_filename}")

    predictions = model.predict(X_test)
    
    return model, X_test, y_test, predictions

def evaluate_model(y_test, predictions):
    """Evaluates the model and prints the metrics."""
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"\nModel Performance:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

def predict_next_day_price(model, stock_data):
    """Predicts the next day's closing price."""
    features = ['Open', 'High', 'Low', 'Volume', 'Moving_Average_50', 'RSI', 'MACD']
    last_row = stock_data[features].iloc[[-1]]
    
    next_day_prediction = model.predict(last_row)
    
    print(f"\nDisclaimer: Stock market prediction is highly speculative.")
    print(f"Predicted closing price for the next trading day: {next_day_prediction[0]:.2f}")

def plot_results(actual, predicted):
    """Plots the actual vs. predicted stock prices."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label='Actual Prices')
    plt.plot(actual.index, predicted, label='Predicted Prices', linestyle='--')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 1. Get User Input
    TICKER = input("Enter the stock ticker (e.g., AAPL): ")
    START_DATE = input("Enter the start date (YYYY-MM-DD): ")
    END_DATE = input("Enter the end date (YYYY-MM-DD): ")

    # 2. Fetch Data
    data = fetch_stock_data(TICKER, START_DATE, END_DATE)

    if not data.empty:
        # 3. Preprocess Data
        processed_data = preprocess_data(data)

        # 4. Train Model and Get Predictions
        model, X_test, y_test, predicted_prices = train_or_load_model(processed_data, TICKER)

        # 5. Evaluate the model
        evaluate_model(y_test, predicted_prices)

        # 6. Predict next day's price
        predict_next_day_price(model, processed_data)

        # 7. Visualize the results
        plot_results(y_test, predicted_prices)