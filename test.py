from stock_trading import StockPredictor
import pandas as pd
import time
import pickle
from datetime import datetime
import sys

ticker = 'solusd'
bartime = '4_hour'
parentdir = 'crypto'
frac_change_lower = -2 # 0.125 for 5_min btc
frac_change_upper = 2
frac_high_upper = 2
frac_low_upper = 2
n_steps_frac_change = 50
n_steps_frac_high = 10
n_steps_frac_low = 10

start_time = time.time()
predictor = StockPredictor(ticker,bartime,parentdir, model_type='gaussian',frac_change_lower=frac_change_lower,frac_change_upper=frac_change_upper,
                           frac_high_upper=frac_high_upper,frac_low_upper=frac_low_upper,n_steps_frac_change=n_steps_frac_change,
                           n_steps_frac_high=n_steps_frac_high,n_steps_frac_low=n_steps_frac_low)
data = pd.read_csv('data/{p}/data/formatted/{t}/{t}_{b}_data_formatted.csv'.format(p=parentdir,t=ticker,b=bartime))
predictor.fit(data, train_start_date=datetime(2023,3,1,3,0,0))
predictor.predict_close_prices_for_days(252,recent=True,with_plot=True,save_plot=True)
predictor.save_pred()
predictor.save_model()
end_time = time.time()
total_runtime = end_time - start_time
print("Total runtime of the function:", total_runtime, "seconds")
    

print(f'Predicted MAPE: {predictor.getMAPE()}')
