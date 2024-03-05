import pickle
import pandas as pd
from pathlib import Path
from StockPredictor import StockPredictor
import logging

class ModelChecker(object):
    def __init__(self):
        self._init_logger()
        # set tqdm barformat
        self.bar_format ='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        
        # load arrays
        self.sp500_tickers = pickle.load(open('iterables/sp500_tickers.pkl', 'rb'))
        self.nasdaq_100_tickers = pickle.load(open('iterables/nasdaq_100_tickers.pkl', 'rb'))
        self.time_intervals = pickle.load(open('iterables/time_intervals.pkl', 'rb'))

    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)


    def pnl_calc(self, data):
        pnl, open_avg = 0, 0
        for idx, row in data.iterrows():
            # open average
            open_avg += row['open']
            pred_diff = row['open'] - row['pred']
            actual_diff = row['open'] - row['close']
            # if trading in same direction 
            if ((pred_diff >= 0 and actual_diff >= 0) or 
                (pred_diff <= 0 and actual_diff <= 0)):
                pnl += abs(actual_diff)
            else: # assumes trade at open
                pnl -= abs(actual_diff)
        open_avg = open_avg / len(data)
        return round(pnl, 2), round(open_avg, 2)

    def calc_all_pnl(self, filepath=None):

        idx_time = {'1_day': 0, '4_hour': 1, '1_hour': 2, 'open_day': 3, 'open_4h': 4, 'open_1h': 5}
        
        pnl_data = {}
        
        for ticker in self.sp500_tickers:
            pnl_row = [0,0,0,0,0,0]
            for bartime in self.time_intervals:
                data = pd.read_csv('data/predicted/{t}/{t}_{b}_pred.csv'.format(t=ticker,b=bartime))
                # pnl for bartime and average open for bartime
                pnl_row[idx_time[bartime]], pnl_row[idx_time[bartime]+3] = self.pnl_calc(data)
            pnl_data[ticker] = pnl_row
        pnl_df = pd.DataFrame.from_dict(pnl_data,orient='index',columns=['1_day', '4_hour', '1_hour', 'open_day', 'open_4h', 'open_1h'])
        if filepath:
            pnl_df.to_csv(filepath)
        return pnl_df
    
    def find_losers(self, bartime, pnl_df):
        df_losers = pnl_df[bartime].loc[pnl_df[bartime] <= 0]
        return df_losers

    def getOldData(self, ticker, bartime):
        return pd.read_csv('data/formatted/{t}/{t}_{b}_data_formatted.csv'.format(t=ticker,b=bartime))

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
        tickers_list = [ticker for ticker, _ in losers.items()]
        while len(tickers_list) > 0:
            ticker = tickers_list.pop()
            stock_predictor = StockPredictor(ticker=ticker, bartime=bartime, load_model=False, model_type='gaussian')
            data = pd.read_csv('data/formatted/{t}/{t}_{b}_data_formatted.csv'.format(t=ticker,b=bartime))
            
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
                stock_predictor.save_model("models/model/{t}/model_{t}_{b}.pkl".format(t=ticker,b=bartime))
                stock_predictor.save_pred("data/predicted/{t}/{t}_{b}_pred.csv".format(t=ticker,b=bartime))
            else:
                # add it back to the queue
                tickers_list.append(ticker)
def main():
    model_check = ModelChecker()
    pnl_df = model_check.calc_all_pnl()
    model_check.recompile_losers('1_day',pnl_df)

    #pnl_data = pd.read_csv('data/pnl/gaussian/pnl_summary.csv')

if __name__=='__main__':
    main()