
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

app = Flask(__name__)
CORS(app)

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def preprocess_data(stock_data):
    stock_data.dropna(inplace=True)
    stock_data['Moving_Average_50'] = stock_data['Close'].rolling(window=50).mean()
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    exp1 = stock_data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = exp1 - exp2
    stock_data.dropna(inplace=True)
    return stock_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        ticker = data['ticker']
        start_date = data['start_date']
        end_date = data['end_date']
        force_retrain = data.get('force_retrain', False)

        stock_data = fetch_stock_data(ticker, start_date, end_date)
        if stock_data.empty:
            return jsonify({'error': 'Could not fetch stock data. Check the ticker or date range.'}), 400

        processed_data = preprocess_data(stock_data)

        if processed_data.empty:
            return jsonify({'error': 'Not enough data to process. Please select a larger date range.'}), 400

        features = ['Open', 'High', 'Low', 'Volume', 'Moving_Average_50', 'RSI', 'MACD']
        target = 'Close'

        X = processed_data[features]
        y = processed_data[target]

        model_filename = f'{ticker}_model.joblib'
        if not force_retrain and os.path.exists(model_filename):
            model = joblib.load(model_filename)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y) # Train on the entire dataset
            joblib.dump(model, model_filename)

        predictions = model.predict(X)

        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)

        last_row = processed_data[features].iloc[[-1]]
        next_day_prediction = model.predict(last_row)

        # Correlation Analysis
        correlation = None
        if ticker.upper() == '^GSPC':
            correlation = 1.0
        else:
            sp500 = fetch_stock_data('^GSPC', start_date, end_date)
            if not sp500.empty:
                df = pd.concat([y, sp500['Close']], axis=1).dropna()
                df.columns = [ticker, '^GSPC']
                if len(df) > 1:
                    returns = df.pct_change()
                    if len(returns) > 1:
                        correlation = returns[ticker].corr(returns['^GSPC'])

        # Volatility Calculation
        daily_returns = y.pct_change()
        volatility = float(daily_returns.std() * np.sqrt(252)) # Annualized volatility

        return jsonify({
            'dates': processed_data.index.strftime('%Y-%m-%d').tolist(),
            'actual': y.values.flatten().tolist(),
            'predicted': predictions.flatten().tolist(),
            'mse': f'{mse:.2f}',
            'r2': f'{r2:.2f}',
            'next_day_prediction': f'{next_day_prediction[0]:.2f}',
            'correlation': f'{float(correlation):.2f}' if correlation is not None else 'N/A',
            'volatility': f'{volatility:.2%}' if volatility is not None else 'N/A'
        })
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/historical-predict', methods=['POST'])
def historical_predict():
    data = request.get_json()
    ticker = data['ticker']
    historical_date_str = data['date']
    historical_date = datetime.strptime(historical_date_str, '%Y-%m-%d')

    # Fetch data up to the historical date
    start_date = historical_date - timedelta(days=365*2) # 2 years of data for training
    end_date_train = historical_date
    end_date_actual = historical_date + timedelta(days=7)
    
    stock_data_hist = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date_actual.strftime('%Y-%m-%d'))

    if stock_data_hist.empty:
        return jsonify({'error': 'Could not fetch historical data.'}), 400

    # Get the actual price for the next day
    next_day_data = stock_data_hist[stock_data_hist.index > end_date_train.strftime('%Y-%m-%d')]
    if next_day_data.empty:
        return jsonify({'error': f'Could not retrieve the actual price for the day after {historical_date_str}'}), 400
    actual_price = next_day_data.iloc[0]['Close']

    training_data = stock_data_hist[stock_data_hist.index <= end_date_train.strftime('%Y-%m-%d')]
    processed_training_data = preprocess_data(training_data.copy())

    if processed_training_data.empty:
        return jsonify({'error': 'Not enough data to train the model for the selected date.'}), 400

    features = ['Open', 'High', 'Low', 'Volume', 'Moving_Average_50', 'RSI', 'MACD']
    target = 'Close'

    X_train = processed_training_data[features]
    y_train = processed_training_data[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    last_row = processed_training_data[features].iloc[[-1]]
    predicted_price = model.predict(last_row)

    return jsonify({
        'predicted_price': f'{predicted_price[0]:.2f}',
        'actual_price': f'{actual_price:.2f}'
    })


if __name__ == '__main__':
    app.run(debug=True)
