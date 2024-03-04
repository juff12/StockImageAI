import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import logging
import pickle

def _init_logger():
    _logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(logging.DEBUG)
    return _logger

def trunc_datetime_month(someDate):
    # arbitrary values for setting all values equal except the month
    return someDate.replace(year=2018, day=1, hour=0, minute=0, second=0, microsecond=0)

def format_data(ticker: str, time_interval: str):
    # returns because there is an issue with the file
    try:
        # load raw data frames
        filepath = Path('data/raw/'+ticker+'/'+ticker+"_"+time_interval+"_data_raw.csv")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(filepath)
        # function to convert Unix msec timestamp to datetime (YYYY-MM-DD)
        convert_date = lambda x: datetime.fromtimestamp(x / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
        df['date'] = df['timestamp'].apply(convert_date)
    except Exception:
        # skip, file formated wrong
        return
    # remove uncessary columns
    df = df[['date','open','high','low','close','volume']]
    
    # out file path
    filepath = Path('data/formatted/'+ticker+'/'+ticker+"_"+time_interval+"_data_formatted.csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)

def main():
    # read in the iterables
    sp500_tickers = pickle.load(open('iterables/sp500_tickers.pkl', 'rb'))
    nasdaq_100_tickers = pickle.load(open('iterables/nasdaq_100_tickers.pkl', 'rb'))
    time_intervals = pickle.load(open('iterables/time_intervals.pkl', 'rb'))
    
    logger = _init_logger()
    logger.info('>>> Begining Stock Data Formatting')
    for ticker in tqdm(sp500_tickers):
        for time_interval in time_intervals:
            format_data(ticker.lower(), time_interval)
    logger.info('Stock Data Formatting Complete <<<')

if __name__ == '__main__':
    main()