import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import sys

class PolygonFormat(object):
    def __init__(self, ticker, bartime, parentdir):
        self._init_logger()

        self.ticker = ticker.lower()
        self.bartime = bartime
        self.parentdir = parentdir
        self.formatted_data = None

    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

    def format_data(self):
        
        self._logger.info('>>> Begining Data Formatting')
        # returns because there is an issue with the file
        try:
            # load raw data frames
            filestring = 'data/{p}/data/raw/{t}/{t}_{b}_data_raw.csv'.format(p=self.parentdir,
                                                                             t=self.ticker,
                                                                             b=self.bartime)
            filepath = Path(filestring)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df = pd.read_csv(filepath)
            # function to convert Unix msec timestamp to datetime (YYYY-MM-DD)
            convert_date = lambda x: datetime.fromtimestamp(x / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
            df['date'] = df['timestamp'].apply(convert_date)
        except Exception:
            # skip, file formated wrong
            self._logger.error('>>> The file could not be opened, check parent directory. {t} and {b} and {p}'.format(t=self.ticker,
                                                                                                                      b=self.bartime,
                                                                                                                      p=self.parentdir))
            return
        self._logger.info('Data Formatting Complete <<<')
        # remove uncessary columns
        self.formatted_data = df[['date','open','high','low','close','volume']]
        
    def save_formatted_data(self, filestring=None):
        # out file path
        if filestring is None:
            filestring = 'data/{p}/data/formatted/{t}/{t}_{b}_data_formatted.csv'.format(p=self.parentdir,
                                                                                         t=self.ticker,
                                                                                         b=self.bartime)
        filepath = Path(filestring)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:         
            self.formatted_data.to_csv(filepath)
        except:
            self._logger.error('The file could not be loaded.')
            sys.exit()
    
    def seperate_weekend_data(self):
        #weekend
        weekend, weekday = [], []
        for idx, row in self.formatted_data.iterrows():
            if datetime.strptime(row['date'], '%Y-%m-%d').weekday() < 5:
                # its a weekday
                weekday.append(row)
            else: # its a weekend
                weekend.append(row)
        return pd.DataFrame(weekday), pd.DataFrame(weekend)