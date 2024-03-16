import pickle
from tqdm import tqdm
from stock_trading import PolygonFormat

def main():
    """
    Main function to format and save crypto data using PolygonFormat class.
    """
    # read in the iterables
    coinbase_tickers = pickle.load(open('data/crypto/iterables/coinbase_tickers.pkl', 'rb'))
    time_intervals = pickle.load(open('data/crypto/iterables/time_intervals.pkl', 'rb'))
    # loop through the tickers and time intervals
    for ticker in tqdm(coinbase_tickers):
        for bartime in time_intervals:
            # create the PolygonFormat object
            formatter = PolygonFormat(ticker.lower(), bartime, 'crypto')
            formatter.format_data()
            formatter.save_formatted_data()

if __name__ == '__main__':
    main()