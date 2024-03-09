import pickle
from stock_trading import PolygonFormat
import tqdm

def main():
    # read in the iterables
    sp500_tickers = pickle.load(open('data/stock/iterables/sp500_tickers.pkl', 'rb'))
    nasdaq_100_tickers = pickle.load(open('data/stock/iterables/nasdaq_100_tickers.pkl', 'rb'))
    time_intervals = pickle.load(open('data/stock/iterables/time_intervals.pkl', 'rb'))
    
    for ticker in tqdm(sp500_tickers):
        for bartime in time_intervals:
            formatter = PolygonFormat(ticker, bartime, 'stock')
            formatter.format_data()
            formatter.save_formatted_data()

if __name__ == '__main__':
    main()