import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    crypto_tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD',
                    'LINK-USD', 'ADA-USD', 'DOT-USD', 'XLM-USD', 'BNB-USD']
    time_intervals = ['1_day']#,'4_hour','1_hour']
    yf_interval = {'1_day': '1d', '4_hour': '4h', '1_hour': '1h'}
    for ticker in crypto_tickers:
        for bartime in time_intervals:
            data = yf.download(ticker, start="2018-01-01", end="2024-01-01",interval=yf_interval[bartime])
            data.reset_index(inplace=True)
            data.columns = map(str.lower, data.columns)
            ticker = ticker.replace('-','_')
            filepath = Path('data/formatted/{t}/{t}_{b}_data_formatted.csv'.format(t=ticker,b=bartime))
            filepath.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(filepath)

if __name__=='__main__':
    main()