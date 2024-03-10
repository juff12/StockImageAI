
import pandas as pd
from hmmlearn.hmm import GaussianHMM, GMMHMM
from sklearn.model_selection import train_test_split
import time
import numpy as np

class StockPredictor(object):
    def __init__(self, data, 
                 model_type='gaussian', n_hidden_states=4, n_latency_days=10,
                 frac_change_lower=-0.1, frac_change_upper=0.1, frac_high_upper=0.1,
                 frac_low_upper=0.1, n_steps_frac_change=50, n_steps_frac_high=10,
                 n_steps_frac_low=10, n_mix=5):
        
        self._data = data
        
        self.n_latency_days = n_latency_days

        self.hmm = GaussianHMM(n_components=n_hidden_states)
        
        self._compute_all_possible_outcomes(frac_change_lower, frac_change_upper, frac_high_upper,
                                            frac_low_upper, n_steps_frac_change, n_steps_frac_high, n_steps_frac_low)
    
    def _split_train_test_data(self, data, test_size):
        _train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
 
        self._train_data = _train_data
        self._test_data = test_data
    
    
    def _extract_features(self,data):
        open = data[:,0]
        close = data[:,1]
        high = data[:,2]
        low = data[:,3]
 
        # Compute the fraction change in close, high and low prices
        # which would be used a feature
        frac_change = (close - open) / open
        frac_high = (high - open) / open
        frac_low = (open - low) / open
 
        return np.column_stack((frac_change, frac_high, frac_low))
    
    def fit(self, data, test_size=0.33):
        self._split_train_test_data(data, test_size)

        feature_vector = StockPredictor._extract_features(self._train_data)

        
        self.hmm.fit(feature_vector)
    
    def _compute_all_possible_outcomes(self, frac_change_lower, frac_change_upper, frac_high_upper,
                                       frac_low_upper, n_steps_frac_change, n_steps_frac_high, n_steps_frac_low):
        frac_change_range = np.linspace(frac_change_lower, frac_change_upper, n_steps_frac_change)
        frac_high_range = np.linspace(0, frac_high_upper, n_steps_frac_high)
        frac_low_range = np.linspace(0, frac_low_upper, n_steps_frac_low)
 
        self._possible_outcomes = np.array(list(itertools.product(frac_change_range, frac_high_range, frac_low_range)))

    def _get_most_probable_outcome(self, day_index):
        previous_data_start_index = max(0, day_index - self.n_latency_days)
        previous_data_end_index = max(0, day_index - 1)
        previous_data = self._test_data[previous_data_end_index: previous_data_start_index,:]
        previous_data_features = self._extract_features(previous_data)

        outcome_score = np.zeros(self._possible_outcomes.size)
        for i, possible_outcome in enumerate(self._possible_outcomes):
            total_data = np.row_stack((previous_data_features, possible_outcome))
            outcome_score[i] = (self.hmm.score(total_data))
        most_probable_outcome = self._possible_outcomes[np.argmax(outcome_score)]

        return most_probable_outcome

    def predict_close_price(self, day_index):
        open_price = self._test_data[day_index, 0]
        predicted_frac_change, _, _ = self._get_most_probable_outcome(day_index)
        
        return open_price * (1 + predicted_frac_change)
    
    def predict_close_prices_for_days(self, days):
        if days > len(self._test_data):
            days = len(test_data)
        predicted_close_prices = np.zeros(days)
        # predict close prices for # days
        for day_index in range(days):
            predicted_close_prices[day_index] = self.predict_close_price(day_index)
        
def main():
    start_time = time.time()
    ticker = 'ethusd'
    bartime = '1_hour'
    parentdir = 'crypto'
    frac_change_lower = -0.1 # 0.125 for 5_min btc
    frac_change_upper = 0.1
    frac_high_upper = 0.1
    frac_low_upper = 0.1
    n_steps_frac_change = 50
    n_steps_frac_high = 10
    n_steps_frac_low = 10
    data = pd.read_csv('data/{p}/data/formatted/{t}/{t}_{b}_data_formatted.csv'.format(p=parentdir,t=ticker,b=bartime))
    data = data[['open','close','high','low']]
    data = data.to_numpy()

    predictor = StockPredictor(data, model_type='gaussian',frac_change_lower=frac_change_lower,frac_change_upper=frac_change_upper,
                            frac_high_upper=frac_high_upper,frac_low_upper=frac_low_upper,n_steps_frac_change=n_steps_frac_change,
                            n_steps_frac_high=n_steps_frac_high,n_steps_frac_low=n_steps_frac_low)
    predictor.fit(data)
    predictor.predict_close_prices_for_days(100)
    end_time = time.time()
    total_runtime = end_time - start_time
    print("Total runtime of the function:", total_runtime, "seconds")
    

    
if __name__=='__main__':
    main()