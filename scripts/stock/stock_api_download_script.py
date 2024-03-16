from polygon import RESTClient
import pandas as pd
from pathlib import Path
import pickle

def get_ticker_data(presets: tuple, client: RESTClient):
    """
    Retrieves ticker data from the API and saves it as a CSV file.

    Args:
        presets (tuple): A tuple containing the preset values for ticker, multiplier, timespan, start, end, and limit.
        client (RESTClient): An instance of the RESTClient class used to make API requests.

    Returns:
        None
    """
    # unpack the presets
    ticker, multiplier, timespan, start, end, limit = presets
    # get the data and load into array
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=multiplier, 
                              timespan=timespan, from_=start, to=end, limit=limit):
        aggs.append(a)

    # convert to dataframe and save as csv
    df = pd.DataFrame(aggs)
    filestring = 'data/stock/data/raw{t}/{t}_{m}_{b}_data_raw.csv'.format(t=str(ticker).lower(),
                                                                          m=str(multiplier),
                                                                          b=timespan)
    filepath = Path(filestring)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)


def main():
    """
    Main function to download stock data using an API.

    Reads in the iterables of stock tickers, connects to the API,
    sets presets for the desired time intervals, and calls the
    `get_ticker_data` function for each ticker and preset.

    Args:
        None

    Returns:
        None
    """
    # read in the iterables
    sp500_tickers = pickle.load(open('data/stock/iterables/sp500_tickers.pkl', 'rb'))
    nasdaq_100_tickers = pickle.load(open('data/stock/iterables/nasdaq_100_tickers.pkl', 'rb'))

    # yapi key
    api_key = ""

    # connect to API
    client = RESTClient(api_key=api_key)

    # presets
    start = '2018-01-01'
    end = '2024-01-01'
    limit = 50000

    # for each desired time interval
    # examples below
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