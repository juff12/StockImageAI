from polygon import RESTClient
import pandas as pd
from pathlib import Path
import pickle
from datetime import date
from tqdm import tqdm

def get_ticker_data(presets: tuple, client: RESTClient):
    """
    Fetches ticker data from a REST API and saves it to a CSV file.

    Args:
        presets (tuple): A tuple containing the preset values for ticker, multiplier, timespan, start, end, and limit.
        client (RESTClient): An instance of the RESTClient class used to make API requests.

    Returns:
        None
    """
    # Extract the values from the presets tuple
    ticker, multiplier, timespan, start, end, limit = presets
    # Format the ticker
    _ticker = "X:" + ticker
    
    # Fetch the aggregated data from the API
    aggs = []
    for a in client.list_aggs(ticker=_ticker, multiplier=multiplier, timespan=timespan, from_=start, to=end, limit=limit):
        aggs.append(a)
    
    df = pd.DataFrame(aggs)
    
    # Define the file path for saving the data
    filestring = 'data/crypto/data/raw/{t}/{t}_{m}_{b}_data_raw.csv'.format(t=str(ticker).lower(), m=str(multiplier), b=timespan)
    filepath = Path(filestring)
    # Create the directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Save the DataFrame to a CSV file
    df.to_csv(filepath)


def main():
    """
    Main function to download ticker data from a cryptocurrency API.

    Reads in the iterables for coinbase tickers and time intervals.
    Connects to the API using the provided API key.
    Sets presets for start date, end date, and data limit.
    Loops through each ticker and time interval to download the ticker data.

    Args:
        None

    Returns:
        None
    """
    # read in the iterables
    coinbase_tickers = pickle.load(open('data/crypto/iterables/coinbase_tickers.pkl', 'rb'))
    time_intervals = pickle.load(open('data/crypto/iterables/time_intervals.pkl', 'rb'))
    
    # api key
    api_key = ""

    # connect to API
    client = RESTClient(api_key=api_key)

    # presets
    start = '2018-01-01'
    end = date.today().strftime('%Y-%m-%d')
    limit = 50000

    # loop through each ticker and time interval
    for ticker in tqdm(coinbase_tickers):
        for bartime in time_intervals:
            multiplier, timespan = bartime.split('_')
            # reformats minute
            if timespan == 'minute':
                timespan = 'minute'
                start = '2020-01-01'
            presets = (ticker,int(multiplier),timespan,start,end,limit)
            get_ticker_data(presets, client)

# run scripts
if __name__ == '__main__':
    main()