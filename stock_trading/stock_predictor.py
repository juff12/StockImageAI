import pickle
import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM, GMMHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
from timeit import default_timer as timer
from datetime import date, datetime, timedelta


# Supress warning in hmmlearn
warnings.filterwarnings("ignore")
# Change plot style to ggplot (for better and more aesthetic visualisation)
plt.style.use('ggplot')

class StockPredictor(object):
    def __init__(self, ticker, bartime, parentdir='', load_model=False, 
                 model_type='gaussian', n_hidden_states=4, n_latency_days=10,
                 frac_change_lower=-0.1, frac_change_upper=0.1, frac_high_upper=0.1,
                 frac_low_upper=0.1, n_steps_frac_change=50, n_steps_frac_high=10,
                 n_steps_frac_low=10, n_mix=5):
        self._init_logger()
        # set tqdm barformat
        self.bar_format ='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        self.ticker = ticker
        self.bartime = bartime
        self.parentdir = parentdir
        self.mape = None
        self.predictions = None
 
        self.n_latency_days = n_latency_days

        # set the model type name
        self.model_type = model_type

        if load_model:
            self.load_model()
        else:
            if self.model_type == 'gmmhmm':
                self.hmm = GMMHMM(n_components=n_hidden_states, n_mix=n_mix)
            else:
                self.hmm = GaussianHMM(n_components=n_hidden_states)
        
        self._compute_all_possible_outcomes(frac_change_lower, frac_change_upper, frac_high_upper,
                                            frac_low_upper, n_steps_frac_change, n_steps_frac_high, n_steps_frac_low)
    
    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

    def load_test(self, test_data):
        self._logger.info('>>> Loading the test data')
        self._test_data=test_data
        self._logger.info('>>> Test data loaded')
        
    def load_model(self, filepath=None):
        self._logger.info('>>> Loading the model')
        if filepath is None:
            filepath = Path('data/{p}/models/{m}/{t}/model_{t}_{b}.pkl'.format(p=self.parentdir,
                                                                               m=self.model_type,
                                                                               t=self.ticker,
                                                                               b=self.bartime))
        try:
            self.hmm = pickle.load(open(filepath, 'rb'))
            self._logger.info('>>> Model Loaded')
        except:
            self._logger.error('The model could not be loaded')

    def _split_train_test_data(self, data, test_size):
        _train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
 
        self._train_data = _train_data
        self._test_data = test_data
 
    @staticmethod
    def _extract_features(data):
        open = np.array(data['open'])
        close = np.array(data['close'])
        high = np.array(data['high'])
        low = np.array(data['low'])
 
        # Compute the fraction change in close, high and low prices
        # which would be used a feature
        frac_change = (close - open) / open
        frac_high = (high - open) / open
        frac_low = (open - low) / open
 
        return np.column_stack((frac_change, frac_high, frac_low))
    
    def fit(self, data: pd.DataFrame, test=True, test_size=0.33):
        self._split_train_test_data(data, test_size)
        self._logger.info('>>> Extracting Features')
        feature_vector = StockPredictor._extract_features(self._train_data)
        self._logger.info('Features extraction Completed <<<')
        
        self.hmm.fit(feature_vector)
        
    def save_model(self, filepath=None):
        # save the model
        if filepath is None:
            filepath = Path("data/{p}/models/{m}/{t}/model_{t}_{b}.pkl".format(p=self.parentdir,
                                                                               m=self.model_type,
                                                                               t=self.ticker,
                                                                               b=self.bartime))
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as file:
            pickle.dump(self.hmm, file)
    
    
    def _compute_all_possible_outcomes(self, frac_change_lower, frac_change_upper, frac_high_upper,
                                       frac_low_upper, n_steps_frac_change, n_steps_frac_high, n_steps_frac_low):
        frac_change_range = np.linspace(frac_change_lower, frac_change_upper, n_steps_frac_change)
        frac_high_range = np.linspace(0, frac_high_upper, n_steps_frac_high)
        frac_low_range = np.linspace(0, frac_low_upper, n_steps_frac_low)
 
        self._possible_outcomes = np.array(list(itertools.product(
            frac_change_range, frac_high_range, frac_low_range)))
    
    def _get_most_probable_outcome(self, day_index):
        previous_data_start_index = max(0, day_index - self.n_latency_days)
        previous_data_end_index = max(0, day_index - 1)
        previous_data = self._test_data.iloc[previous_data_end_index: previous_data_start_index]
        previous_data_features = StockPredictor._extract_features(previous_data)
 
        outcome_score = []
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack((previous_data_features, possible_outcome))
            outcome_score.append(self.hmm.score(total_data))
        most_probable_outcome = self._possible_outcomes[np.argmax(outcome_score)]
 
        return most_probable_outcome

    def predict_close_price(self, day_index):
        open_price = self._test_data.iloc[day_index]['open']
        predicted_frac_change, _, _ = self._get_most_probable_outcome(day_index)
        
        return open_price * (1 + predicted_frac_change)

    def predict_close_prices_for_days(self, days, recent=False, with_plot=False, save_plot=False):
        if self._test_data is None:
            self._logger.error('No test data has been loaded')
            return None
        
        temp_test = self._test_data
        if days > len(self._test_data):
            days = len(self._test_data)
        elif recent: # set the test data to the most recent data
            self._test_data = self._test_data[len(self._test_data)-days:]
        
        self._logger.info('>>> Begining Prediction Generation')
        predicted_close_prices = []
        for day_index in tqdm(range(days),bar_format=self.bar_format,desc='Predictions'):
            predicted_close_prices.append(self.predict_close_price(day_index))
        self._logger.info('Prediction Generation Complete <<<')
        
        test_data = self._test_data[0: days]
        actual_close_prices = test_data['close']
        
        self.predictions = predicted_close_prices
        self.actual_close_data = test_data
        
        self.mape = self.compute_mape(days, predicted_close_prices, actual_close_prices.values)
        
        if with_plot:
            days = np.array(test_data['date'], dtype="datetime64[ms]")

            fig = plt.figure(figsize=(24,12))
            
            axes = fig.add_subplot(111)
            axes.plot(days, actual_close_prices, 'bo-', markersize=3, label="actual")
            axes.plot(days, predicted_close_prices, 'ro-',markersize=3, label="predicted")
            axes.set_title(f'{self.ticker.upper()} - Predicted vs Actual Prices ({self.bartime})')
            axes.set_xlabel('Date')
            axes.set_ylabel('Stock Value (US Dollars)')
            fig.autofmt_xdate()

            plt.legend()
            if save_plot:
                filepath = Path('data/{p}/charts/{m}_charts/{t}/{t}_{b}_chart.png'.format(p=self.parentdir,
                                                                                          m=self.model_type,
                                                                                          t=self.ticker,
                                                                                          b=self.bartime))
                filepath.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(filepath)
            else:
                plt.show()
        
        # change test data back
        self._test_data = temp_test
        
        
        
    def compute_mape(self, days, predicted_close_prices, actual_close_prices):
        if days > len(self._test_data):
            days = len(self._test_data)
        
        mape = []
        
        for day in range(0,days):
            p, y = predicted_close_prices[day], actual_close_prices[day]
            diff_percent = (abs(p - y) / abs(y)) * 100
            mape.append(diff_percent)
        mape = sum(mape) / days
        
        return round(mape, 2)
    
    def save_pred(self, filepath=None):
        df = self.actual_close_data[['date','open','high','low','close']]
        df.loc[:,'pred'] = [round(pred, 2) for pred in self.predictions]
        if filepath is None:
            filepath = Path("data/{p}/predicted/{m}/{t}/{t}_{b}_pred.csv".format(p=self.parentdir,
                                                                                 m=self.model_type,
                                                                                 t=self.ticker,
                                                                                 b=self.bartime))
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)

    def predict_next_timeframe_price(self, data):
        self._test_data = data
        return self.predict_close_price(len(self._test_data) - 1)
    
    def getPredictions(self):
        df = self.actual_close_data[['date','open','high','low','close']]
        df.loc[:,'pred'] = [round(pred, 2) for pred in self.predictions]
        return df

    def getMAPE(self):
        return self.mape