from polygon import RESTClient
import pandas as pd
from pathlib import Path
import pickle

def get_ticker_data(presets: tuple, client: RESTClient):
    ticker, multiplier, timespan, start, end, limit = presets
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=multiplier, 
                              timespan=timespan, from_=start, to=end, limit=limit):
        aggs.append(a)

    df = pd.DataFrame(aggs)
    filestring = 'stock/data/raw{t}/{t}_{m}_{b}_data_raw.csv'.format(t=str(ticker).lower(),
                                                                     m=str(multiplier),
                                                                     b=timespan)
    filepath = Path(filestring)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)

# code of scipt
def main():
    # read in the iterables
    sp500_tickers = pickle.load(open('stock/iterables/sp500_tickers.pkl', 'rb'))
    nasdaq_100_tickers = pickle.load(open('stock/iterables/nasdaq_100_tickers.pkl', 'rb'))

    # yapi key
    api_key = ""

    # connect to API
    client = RESTClient(api_key=api_key)

    # presets
    start = '2018-01-01'
    end = '2024-01-01'
    limit = 50000

    # daily data
    for ticker in sp500_tickers:
        presets = (ticker,1,'day',start,end,limit)
        get_ticker_data(presets, client)
        presets = (ticker,1,'hour',start,end,limit)
        get_ticker_data(presets, client)
        presets = (ticker,4,'hour',start,end,limit)
        get_ticker_data(presets, client)

# run scripts
if __name__ == '__main__':
    main()