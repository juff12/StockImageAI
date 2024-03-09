import pickle
import pandas as pd
from pathlib import Path
from .stock_predictor import StockPredictor
import logging
import numpy as np

class ModelChecker(object):
    def __init__(self, ticker_list, time_intervals, 
                 parentdir='', model_type='gaussian'):
        self._init_logger()
        # set tqdm barformat
        self.bar_format ='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        self.ticker_list = ticker_list
        self.time_intervals = time_intervals
        self.parentdir = parentdir
        self.model_type = model_type

    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

    def pnl_calc(self, data):
        pnl, open_avg = [], 0
        for idx, row in data.iterrows():
            # open average
            open_avg += row['open']
            pred_diff = row['open'] - row['pred']
            actual_diff = row['open'] - row['close']
            # if trading in same direction 
            if ((pred_diff >= 0 and actual_diff >= 0) or 
                (pred_diff <= 0 and actual_diff <= 0)):
                pnl.append(abs(actual_diff))
            else: # assumes trade at open
                pnl.append(-1*abs(actual_diff))
        open_avg = open_avg / len(data)
        pnl = np.array(pnl)
        mean, std, var = np.mean(pnl), np.std(pnl), np.var(pnl)
        pnl = np.sum(pnl)        
        return [round(item,2) for item in [pnl, open_avg, mean, std, var]]

    def calc_all_pnl(self, filepath=None):
        metrics = ['_pnl', '_open_avg', '_mean', '_std', '_var']
        labels = [bartime + metric for bartime in self.time_intervals for metric in metrics]
        
        pnl_data = {}
        
        for ticker in self.ticker_list:
            pnl_row = []
            for bartime in self.time_intervals:
                data = pd.read_csv("{p}/data/predicted/{m}/{t}/{t}_{b}_pred.csv".format(p=self.parentdir,
                                                                                        m=self.model_type,
                                                                                        t=ticker,
                                                                                        b=bartime))
                # pnl for bartime and average open for bartime
                pnl_row.append(self.pnl_calc(data))
            pnl_data[ticker] = np.array(pnl_row).flatten()
        pnl_df = pd.DataFrame.from_dict(pnl_data,orient='index',columns=labels)
        if filepath:
            pnl_df.to_csv(filepath)
        return pnl_df
    
    def find_losers(self, bartime, pnl_df):
        df_losers = pnl_df[bartime].loc[pnl_df[bartime] <= 0]
        return df_losers

    def getOldData(self, ticker, bartime):
        return pd.read_csv("{p}/data/predicted/{m}/{t}/{t}_{b}_pred.csv".format(p=self.parentdir,
                                                                                m=self.model_type,
                                                                                t=ticker,
                                                                                b=bartime))

    def getRecentData(self, ticker, bartime):
        # TO DO:
        # make api call for new data
        return None
    
    def dataWindowShift(self, ticker, bartime, window):
        data = self.getRecentData(ticker, bartime)
        return data[window:]

    def recompile_strategy_selector(self, ticker, bartime, attempt):
        attempt_dict = {0: self.getOldData(ticker, bartime),
                        1: self.getRecentData(ticker, bartime), 
                        2: self.dataWindowShift(ticker, bartime, 100),
                        3: self.dataWindowShift(ticker, bartime, 300),
                        4: self.dataWindowShift(ticker, bartime, 400)
                        }


    def recompile_losers(self, bartime, pnl_df):
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