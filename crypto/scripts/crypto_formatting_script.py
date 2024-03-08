import pickle
import sys
sys.path
from PolygonFormat import PolygonFormat
import tqdm

def main():
    # read in the iterables
    coinbase_tickers = pickle.load(open('crypto/iterables/coinbase_tickers.pkl', 'rb'))
    time_intervals = pickle.load(open('crypto/iterables/time_intervals.pkl', 'rb'))
    
    for ticker in tqdm(coinbase_tickers):
        for bartime in time_intervals:
            formatter = PolygonFormat(ticker, bartime, 'crypto')
            formatter.format_data()

if __name__ == '__main__':
    main()