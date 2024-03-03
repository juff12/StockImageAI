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
import sys

# Supress warning in hmmlearn
warnings.filterwarnings("ignore")
# Change plot style to ggplot (for better and more aesthetic visualisation)
plt.style.use('ggplot')

class StockPredictor(object):
    def __init__(self, company, load_model=False, model_type='gaussian',
                 test_size=0.33, n_hidden_states=4, n_latency_days=10,
                 n_steps_frac_change=50, n_steps_frac_high=10,
                 n_steps_frac_low=10, n_mix=5):
        self._init_logger()
        # set tqdm barformat
        self.bar_format ='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        self.company = company[0]
        self.bartime = company[1]
        
        # sets default test size
        self.test_size = test_size
        
        self.n_latency_days = n_latency_days

        # set the model type name
        self.model_type = model_type
        
        if load_model is True:
            self.load_model()
        else:
            if self.model_type == 'gmmhmm':
                self.hmm = GMMHMM(n_components=n_hidden_states, n_mix=n_mix)
            else:
                self.hmm = GaussianHMM(n_components=n_hidden_states)
        
        self._compute_all_possible_outcomes(
            n_steps_frac_change, n_steps_frac_high, n_steps_frac_low)
    
    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

    def load_model(self):
        self._logger.info('>>> Loading the model')
        filepath = Path('models/{m}/{t}/model_{t}_{b}.pkl'.format(m=self.model_type,
                                                                  t=self.company,
                                                                  b=self.bartime))
        try:
            self.hmm = pickle.load(open(filepath),'rb')
        except:
            self._logger.error('The model could not be loaded')
            sys.exit()
        self._logger.info('>>> Model Loaded')

    def _split_train_test_data(self, test_size):
        data = pd.read_csv(
            'data/formatted/{c}/{c}_{b}_data_formatted.csv'.format(c=self.company,b=self.bartime))
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
    
    def fit(self):
        self._split_train_test_data(self.test_size)
        self._logger.info('>>> Extracting Features')
        feature_vector = StockPredictor._extract_features(self._train_data)
        self._logger.info('Features extraction Completed <<<')
        
        self.hmm.fit(feature_vector)
        
        # save the model
        filepath = Path("models/model/{c}/model_{c}_{b}.pkl".format(c=self.company,b=self.bartime))
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as file:
            pickle.dump(self.hmm, file)
    
    
    def _compute_all_possible_outcomes(self, n_steps_frac_change,
                                       n_steps_frac_high, n_steps_frac_low):
        frac_change_range = np.linspace(-0.1, 0.1, n_steps_frac_change)
        frac_high_range = np.linspace(0, 0.1, n_steps_frac_high)
        frac_low_range = np.linspace(0, 0.1, n_steps_frac_low)
 
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

    def predict_close_prices_for_days(self, days, with_plot=False):
        start = timer()
        if days > len(self._test_data):
            days = len(self._test_data)
        self._logger.info('>>> Begining Prediction Generation')
        predicted_close_prices = []
        for day_index in tqdm(range(days),bar_format=self.bar_format,desc='Predictions'):
            predicted_close_prices.append(self.predict_close_price(day_index))
        self._logger.info('Prediction Generation Complete <<<')
        
        test_data = self._test_data[0: days]
        actual_close_prices = test_data['close']
        self.pred_save(predicted_close_prices, test_data)

        mape = self.compute_mape(days, predicted_close_prices, actual_close_prices.values)
        self._logger.info(timer() - start)
        
        if with_plot:
            days = np.array(test_data['date'], dtype="datetime64[ms]")

            fig = plt.figure(figsize=(24,12))
 
            axes = fig.add_subplot(111)
            axes.plot(days, actual_close_prices, 'bo-', markersize=3, label="actual")
            axes.plot(days, predicted_close_prices, 'ro-',markersize=3, label="predicted")
            axes.set_title(f'{self.company.upper()} - Predicted vs Actual Prices ({self.bartime})')
            axes.set_xlabel('Date')
            axes.set_ylabel('Stock Value (US Dollars)')
            fig.autofmt_xdate()

            plt.legend()
            filepath = Path('data/HMM_charts/{c}/{c}_{b}_chart.png'.format(c=self.company,b=self.bartime))
            filepath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath)
            
        return mape
    
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
    
    def pred_save(self, predictions, df):
        df = df[['date','open','high','low','close']]
        df.loc[:,'pred close'] = [round(pred, 2) for pred in predictions]
        filepath = Path(f"data/predicted/{self.company}/{self.company}_{self.bartime}_pred.csv")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)