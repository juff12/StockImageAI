import pickle
import pandas as pd
from pathlib import Path
from .stock_predictor import StockPredictor
import logging
import numpy as np

class ModelChecker(object):
    """
    A class for checking and analyzing trading models.

    Args:
        ticker_list (list): A list of ticker symbols.
        time_intervals (list): A list of time intervals.
        parentdir (str, optional): The parent directory path. Defaults to ''.
        model_type (str, optional): The type of trading model. Defaults to 'gaussian'.
    """

    def __init__(self, ticker_list, time_intervals, parentdir='', model_type='gaussian'):
        self._init_logger()
        # set tqdm barformat
        self.bar_format ='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        self.ticker_list = ticker_list
        self.time_intervals = time_intervals
        self.parentdir = parentdir
        self.model_type = model_type

    def _init_logger(self):
        """
        Initialize the logger for logging debug messages.
        """
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

    def pnl_calc(self, data):
        """
        Calculate the profit and loss (pnl) for a given dataset.

        Args:
            data (pd.DataFrame): The dataset containing open, pred, and close prices.

        Returns:
            list: A list containing pnl, open_avg, mean, std, and var.
        """
        pnl, open_avg = [], 0
        for idx, row in data.iterrows():
            # open average
            open_avg += row['open']
            pred_diff = row['open'] - row['pred']
            actual_diff = row['open'] - row['close']
            # if trading in same direction 
            if ((pred_diff >= 0 and actual_diff >= 0) or 
                (pred_diff <= 0 and actual_diff <= 0)):
                pnl.append(abs(actual_diff) / row['open'])
            else: # assumes trade at open
                pnl.append(-1*abs(actual_diff) / row['open'])
        open_avg = open_avg / len(data)
        pnl = np.array(pnl)
        mean, std, var = np.mean(pnl), np.std(pnl), np.var(pnl)
        pnl = np.sum(pnl)
        return [pnl, open_avg, mean, std, var]

    def calc_all_pnl(self, save_steps=True):
        """
        Calculate pnl for all time intervals and save the results to CSV files.

        Args:
            save_steps (bool, optional): Whether to save intermediate steps. Defaults to True.
        """
        metrics = ['_pnl', '_open_avg', '_mean', '_std', '_var']
        
        # iterate through tickers and time intervals
        for bartime in self.time_intervals:
            pnl_time = [] # pnl for each time frame
            for ticker in self.ticker_list:
                # read in the predicted data
                data = pd.read_csv("data/{p}/predicted/{m}/{t}/{t}_{b}_pred.csv".format(p=self.parentdir,
                                                                                        m=self.model_type,
                                                                                        t=ticker,
                                                                                        b=bartime))
                # pnl for bartime and average open for bartime
                pnl_time.append([ticker] + self.pnl_calc(data))
            # save steps
            if save_steps:
                # create a temp dataframe
                labels = ["ticker"] + [bartime + metric for metric in metrics]
                df = pd.DataFrame(pnl_time, columns=labels)
                df.set_index('ticker', inplace=True)
                # create filepath
                filepath = Path('data/{p}/pnl/{m}/pnl_{b}_summary.csv'.format(p=self.parentdir,
                                                                              m=self.model_type,
                                                                              t=ticker,
                                                                              b=bartime))
                # create the directory if it does not exist
                filepath.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(filepath)
    
    def find_losers(self, bartime, pnl_df):
        """
        Find the tickers with negative pnl for a given time interval.

        Args:
            bartime (str): The time interval.
            pnl_df (pd.DataFrame): The dataframe containing pnl data.

        Returns:
            pd.Series: The tickers with negative pnl.
        """
        df_losers = pnl_df[bartime].loc[pnl_df[bartime] <= 0]
        return df_losers

    def getOldData(self, ticker, bartime):
        """
        Get the old data for a specific ticker and time interval.

        Args:
            ticker (str): The ticker symbol.
            bartime (str): The time interval.

        Returns:
            pd.DataFrame: The old data.
        """
        return pd.read_csv("{p}/data/predicted/{m}/{t}/{t}_{b}_pred.csv".format(p=self.parentdir,
                                                                                m=self.model_type,
                                                                                t=ticker,
                                                                                b=bartime))

    def getRecentData(self, ticker, bartime):
        """
        Get the recent data for a specific ticker and time interval.

        Args:
            ticker (str): The ticker symbol.
            bartime (str): The time interval.

        Returns:
            None: Placeholder for making an API call to get new data.
        """
        # TO DO:
        # make api call for new data
        return None
    
    def dataWindowShift(self, ticker, bartime, window):
        """
        Shift the data window for a specific ticker and time interval.

        Args:
            ticker (str): The ticker symbol.
            bartime (str): The time interval.
            window (int): The size of the data window.

        Returns:
            pd.DataFrame: The shifted data.
        """
        data = self.getRecentData(ticker, bartime)
        return data[window:]

    def recompile_strategy_selector(self, ticker, bartime, attempt):
        """
        Recompile the strategy selector for a specific ticker and time interval.

        Args:
            ticker (str): The ticker symbol.
            bartime (str): The time interval.
            attempt (int): The attempt number.

        Returns:
            None: Placeholder for recompiling the strategy selector.
        """
        attempt_dict = {0: self.getOldData(ticker, bartime),
                        1: self.getRecentData(ticker, bartime), 
                        2: self.dataWindowShift(ticker, bartime, 100),
                        3: self.dataWindowShift(ticker, bartime, 300),
                        4: self.dataWindowShift(ticker, bartime, 400)
                        }


    def recompile_losers(self, bartime, pnl_df):
        """
        Recompile the losers for a specific time interval.

        Args:
            bartime (str): The time interval.
            pnl_df (pd.DataFrame): The dataframe containing pnl data.

        Returns:
            None: Placeholder for recompiling the losers.
        """
        losers = self.find_losers(bartime, pnl_df)
        loser_tickers = [ticker for ticker, _ in losers.items()]
        while len(loser_tickers) > 0:
            ticker = loser_tickers.pop()
            stock_predictor = StockPredictor(ticker=ticker, bartime=bartime, parentdir=self.parentdir, load_model=False, model_type='gaussian')
            data = pd.read_csv(pd.read_csv("{p}/data/predicted/{m}/{t}/{t}_{b}_pred.csv".format(p=self.parentdir,
                                                                                                m=self.model_type,
                                                                                                t=ticker,
                                                                                                b=bartime)))
            
            ######################### 
            # implemenet a data modifier that shifts the window of the data for training
            # train on more recent data
            # keep track of strategies used, iterate through list of strategies
            # strategy_selector()
            stock_predictor.fit(data[len(data) - 800:])
            
            
            stock_predictor.predict_close_prices_for_days(252, with_plot=False, save_plot=False)
            preds = stock_predictor.getPredictions()

            # check if the pnl is positive, discard new open
            new_pnl, _ = self.pnl_calc(preds)
            if new_pnl >= 0:
                stock_predictor.save_model()
                stock_predictor.save_pred()
            else:
                # add it back to the queue
                loser_tickers.append(ticker)