from hmm_training import StockPredictor
import pickle
from pathlib import Path
import pandas as pd
import glob
import os

# read in the iterables
sp500_tickers = pickle.load(open('iterables/sp500_tickers.pkl', 'rb'))
nasdaq_100_tickers = pickle.load(open('iterables/nasdaq_100_tickers.pkl', 'rb'))
time_intervals = pickle.load(open('iterables/time_intervals.pkl', 'rb'))

for ticker in sp500_tickers:
    for bartime in time_intervals:
        for filename in glob.iglob('data/predicted/{t}/{t}_{b}_pred_formatted.csv'.format(t=ticker,b=bartime)):
            os.remove(filename)