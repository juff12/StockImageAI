import pickle
from pathlib import Path
import pandas as pd
import yfinance as yf
import PolygonFormat as PolygonFormat


# read in the iterables
sp500_tickers = pickle.load(open('iterables/sp500_tickers.pkl', 'rb'))
nasdaq_100_tickers = pickle.load(open('iterables/nasdaq_100_tickers.pkl', 'rb'))
time_intervals = pickle.load(open('iterables/time_intervals.pkl', 'rb'))

data = yf.download('aapl', start='2024-01-01', end='2024-03-04',interval='4h')
print(data)