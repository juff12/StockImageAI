import pickle
from pathlib import Path
import pandas as pd
import yfinance as yf
from stock_trading import PolygonFormat


# read in the iterables
sp500_tickers = pickle.load(open('data/stock/iterables/sp500_tickers.pkl', 'rb'))
nasdaq_100_tickers = pickle.load(open('data/stock/iterables/nasdaq_100_tickers.pkl', 'rb'))
time_intervals = pickle.load(open('data/stock/iterables/time_intervals.pkl', 'rb'))

data = yf.download('aapl', start='2024-01-01', end='2024-03-04',interval='4h')
print(data)