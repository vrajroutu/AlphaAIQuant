import os
import time
import argparse

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import schedule

from datetime import datetime, time as dt_time, timedelta
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from openai import AzureOpenAI
from langchain_community.utilities import BingSearchAPIWrapper
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Slack configuration
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')  
SLACK_CHANNEL = os.getenv('SLACK_CHANNEL')  

BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search" 
BING_SUBSCRIPTION_KEY = os.getenv('BING_SUBSCRIPTION_KEY')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')  
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')  
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT')  
AZURE_OPENAI_API_VERSION = '2023-12-01-preview'

# Initialize Slack client
slack_client = WebClient(token=SLACK_BOT_TOKEN)

# Initialize OpenAI client
openai_service_client = AzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_key=AZURE_OPENAI_API_KEY
)

# List of tickers to analyze
tickers = ['TSLA']

# Dictionary to track sent alerts
sent_alerts = {ticker: None for ticker in tickers}

# Function to send Slack alerts
def send_slack_message(message, chart_path=None):
    try:
        slack_client.conversations_join(channel=SLACK_CHANNEL)
        response = slack_client.chat_postMessage(
            channel=SLACK_CHANNEL,
            text=message
        )
        if chart_path:
            response = slack_client.files_upload_v2(
                channels=SLACK_CHANNEL,
                file=chart_path,
                title="Stock Chart"
            )
        print(f"Slack message sent: {message}")  # Debugging statement
    except SlackApiError as e:
        print(f"Error sending message: {e.response['error']}")

# Function to fetch market news
def fetch_market_news(ticker):
    search = BingSearchAPIWrapper(bing_subscription_key=BING_SUBSCRIPTION_KEY,
                                  bing_search_url=BING_SEARCH_URL,
                                  k=5)
    query = f"{ticker} stock news"
    results = search.results(query, 5)
    news_snippets = [result['snippet'] for result in results]
    return news_snippets

# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    sentiment = analysis.sentiment.polarity
    return sentiment

# Function to fetch and analyze sentiment
def fetch_and_analyze_sentiment(ticker):
    news_snippets = fetch_market_news(ticker)
    sentiment_scores = [analyze_sentiment(snippet) for snippet in news_snippets]
    average_sentiment = np.mean(sentiment_scores)
    return average_sentiment

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to train machine learning model
def train_ml_model(data):
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    features = ['MA20', 'MA50', 'RSI', 'MACD', 'Signal Line']
    data = data.dropna(subset=features + ['Target'])
    
    if len(data) < 10:  # Ensure we have enough data for training
        print("Not enough data to train the model.")
        return None
    
    X = data[features]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if len(X_train) == 0 or len(X_test) == 0:  # Ensure non-empty training and test sets
        print("Not enough data to create train/test splits.")
        return None
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    return model

# Function to predict with machine learning model
def predict_with_ml_model(model, data):
    features = ['MA20', 'MA50', 'RSI', 'MACD', 'Signal Line']
    data = data.dropna(subset=features)
    
    if model is None or data.empty:  # Ensure model is trained and data is available
        return []
    
    X = data[features]
    predictions = model.predict(X)
    return predictions

# Function to fetch and analyze stock data
def fetch_and_analyze_stock():
    now = datetime.now().time()
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    after_market = now > market_close or now < market_open

    interval = '1mo' if after_market else '1h'
    period = '5d' if after_market else '1d'

    for ticker in tickers:
        data = yf.download(ticker, period=period, interval=interval)

        if data.empty:
            continue

        data = data.dropna()

        if data.empty:
            continue

        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = calculate_rsi(data)
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

        model = train_ml_model(data)
        ml_predictions = predict_with_ml_model(model, data)

        sentiment_score = fetch_and_analyze_sentiment(ticker)
        combined_predictions = []
        for i in range(len(ml_predictions)):
            if ml_predictions[i] == 1 and sentiment_score > 0:
                combined_predictions.append('Buy Call')
            elif ml_predictions[i] == 0 and sentiment_score < 0:
                combined_predictions.append('Buy Put')
            else:
                combined_predictions.append('Hold')

        plt.figure(figsize=(14, 10))
        plt.subplot(3, 1, 1)
        plt.plot(data['Close'], label='Close Price')
        plt.plot(data['MA20'], label='20-Day MA')
        plt.plot(data['MA50'], label='50-Day MA')
        plt.title(f'{ticker} Stock Price and Moving Averages')
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(data['RSI'], label='RSI')
        plt.axhline(70, color='red', linestyle='--')
        plt.axhline(30, color='green', linestyle='--')
        plt.title('Relative Strength Index (RSI)')
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(data['MACD'], label='MACD')
        plt.plot(data['Signal Line'], label='Signal Line')
        plt.title('MACD')
        plt.legend()
        plt.tight_layout()
        chart_path = f'stock_chart_{ticker}.png'
        plt.savefig(chart_path)
        plt.close()

        for date, prediction in zip(data.index, combined_predictions):
            if prediction in ['Buy Call', 'Buy Put'] and sent_alerts[ticker] != prediction:
                expiry_date = (date + timedelta(days=30)).strftime('%Y-%m-%d')
                send_slack_message(f"{prediction} Alert for {ticker} on {date.strftime('%Y-%m-%d %H:%M:%S')}\nPrediction: {prediction}\nOption Expiry: {expiry_date}", chart_path)
                sent_alerts[ticker] = prediction

# Function to perform research analysis and send to Slack
def perform_research_analysis():
    now = datetime.now().time()
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    after_market = now > market_close or now < market_open

    interval = '1d' if after_market else '1m'
    period = '5d' if after_market else '1d'

    for ticker in tickers:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            continue

        data = data.dropna()
        if data.empty:
            continue

        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = calculate_rsi(data)
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

        news_snippets = fetch_market_news(ticker)
        news_summary = "\n".join(news_snippets)

        def predict_stock_movement(data):
            prompt = f"""
            Given the following stock data for {ticker}, predict if the stock will move up or down based on the following technical analysis principles:
            - A 'Buy Call' signal is generated when the stock's closing price is above the 20-day moving average, the RSI is below 30, and the MACD is above the Signal Line.
            - A 'Buy Put' signal is generated when the stock's closing price is below the 20-day moving average, the RSI is above 70, and the MACD is below the Signal Line.
            - The 20-day moving average is a short-term trend indicator.
            - The 50-day moving average is a medium-term trend indicator.
            - The RSI (Relative Strength Index) measures the speed and change of price movements and is used to identify overbought or oversold conditions.
            - The MACD (Moving Average Convergence Divergence) is used to identify changes in the strength, direction, momentum, and duration of a trend.

            Here is the recent stock data:
            {data.tail(10).to_string()}

            Here are the latest news headlines:
            {news_summary}

            Based on the above principles, data, news, and model predictions, provide a prediction:
            """
            print(f"OpenAI prompt: {prompt}")  # Debugging statement
            completion = openai_service_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            prediction = completion.choices[0].message.content.strip()
            print(f"OpenAI prediction: {prediction}")  # Debugging statement
            return prediction

        prediction = predict_stock_movement(data)

        plt.figure(figsize=(14, 10))
        plt.subplot(3, 1, 1)
        plt.plot(data['Close'], label='Close Price')
        plt.plot(data['MA20'], label='20-Day MA')
        plt.plot(data['MA50'], label='50-Day MA')
        plt.title(f'{ticker} Stock Price and Moving Averages')
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(data['RSI'], label='RSI')
        plt.axhline(70, color='red', linestyle='--')
        plt.axhline(30, color='green', linestyle='--')
        plt.title('Relative Strength Index (RSI)')
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(data['MACD'], label='MACD')
        plt.plot(data['Signal Line'], label='Signal Line')
        plt.title('MACD')
        plt.legend()
        plt.tight_layout()
        chart_path = f'stock_chart_{ticker}.png'
        plt.savefig(chart_path)
        plt.close()

        send_slack_message(f"Research Analysis for {ticker}:\n{prediction}", chart_path)


# Schedule the function to run every minute during market hours
def run_scheduler(ignore_market_hours):
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    pre_market_open = dt_time(4, 0)
    after_market_close = dt_time(20, 0)

    def is_market_open():
        now = datetime.now().time()
        return (pre_market_open <= now <= market_close) or (market_close < now <= after_market_close)

    def job():
        if ignore_market_hours or is_market_open():
            fetch_and_analyze_stock()

    schedule.every().minute.do(job)
    schedule.every().day.at("09:30").do(perform_research_analysis)
    schedule.every().day.at("12:00").do(perform_research_analysis)
    schedule.every().day.at("20:52").do(perform_research_analysis)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Analysis Script')
    parser.add_argument('--ignore-market-hours', action='store_true', help='Ignore market hours and run the script')
    args = parser.parse_args()

    run_scheduler(args.ignore_market_hours)
