import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(ticker):
    df = yf.download(ticker, start="2015-01-01", end="2020-01-01")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(method='ffill')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled