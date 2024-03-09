import os
from pathlib import Path
from tqdm import tqdm
import pickle
from chart_image_create import screenshot_chart

def main():
    # read in the arrays for tickers
    sp500_tickers = pickle.load(open('stock/iterables/sp500_tickers.pkl', 'rb'))
    nasdaq_100_tickers = pickle.load(open('stock/iterables/nasdaq_100_tickers.pkl', 'rb'))
    time_intervals = pickle.load(open('stock/iterables/time_intervals.pkl', 'rb'))
    
    crypto_tickers = pickle.load(open('crypto/iterables/coinbase_tickers.pkl', 'rb'))
    
    time_intervals = pickle.load(open('crypto/iterables/time_intervals.pkl', 'rb'))
    
    for ticker in tqdm(sp500_tickers, desc='Crypto: '):
        for bartime in time_intervals:
            
            # #if bartime == '5_min' or bartime == '15_min':
            #     continue
            ticker = ticker.lower()
            screenshot_chart(ticker,bartime,'candle','stock')

# run script
if __name__ == '__main__':
    main()