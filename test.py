from stock_trading import CryptoPredictor
from stock_trading import CoinbaseWrapper
import pandas as pd
import time
import pickle
from datetime import datetime
import sys

coinbase_wrapper = CoinbaseWrapper('btcusd')
btcusd_candles = coinbase_wrapper.fetch_last_n_candles('1_hour', 300)
print(btcusd_candles)
pickle.dump(btcusd_candles, open('coinbase_now.pkl','wb'))

# ticker = 'solusd'
# bartime = '4_hour'
# parentdir = 'crypto'
# frac_change_lower = -0.025 # 0.125 for 5_min btc
# frac_change_upper = 0.1
# n_steps_frac_change = 50

# predictor = CryptoPredictor(ticker,bartime,parentdir, model_type='gaussian',frac_change_lower=frac_change_lower,frac_change_upper=frac_change_upper,
#                            n_steps_frac_change=n_steps_frac_change)
# data = pd.read_csv('data/{p}/data/formatted/{t}/{t}_{b}_data_formatted.csv'.format(p=parentdir,t=ticker,b=bartime))
# predictor.fit(data, train_start_date=datetime(2023,5,1),test_size=0.2)
# predictor.predict_close_prices_for_days(50,recent=True,with_plot=True,save_plot=False)
# predictor.save_pred()
# print(f'Predicted MAPE: {predictor.getMAPE()}')
