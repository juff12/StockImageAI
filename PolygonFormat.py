import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import logging
import pickle

class PolygonFormat(object):
    def __init__(self, ticker, bartime, parentdir):
        self._init_logger()

        self.ticker = ticker
        self.bartime = bartime
        self.parentdir = parentdir

    def _init_logger():
        _logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
        _logger.setLevel(logging.DEBUG)
        return _logger

    def format_data(self):
        
        self.logger.info('>>> Begining Stock Data Formatting')
        # returns because there is an issue with the file
        try:
            # load raw data frames
            filestring = '{p}/data/raw/{t}/{t}_{b}_data_raw.csv'.format(p=self.parentdir,
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
            self.logger.error(f'>>> The file could not be opened, check 
                              parent directory. {self.tickercekr} and 
                              {self.bartime} and {self.parentdir}')
            return
        self.logger.info('Stock Data Formatting Complete <<<')
        # remove uncessary columns
        df = df[['date','open','high','low','close','volume']]
        
        # out file path
        filestring = '{p}/data/formatted/{t}/{t}_{b}_data_formatted.csv'.format(p=self.parentdir,
                                                                                t=self.ticker,
                                                                                b=self.bartime)
        filepath = Path(filestring)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath)