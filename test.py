from stock_trading import StockPredictor
import pandas as pd

ticker = 'btcusd'
bartime = '1_day'
parentdir = 'crypto'
frac_change_lower = -0.125 # 0.125 for 5_min btc
frac_change_upper = 0.125
frac_high_upper = 0.125
frac_low_upper = 0.125
n_steps_frac_change = 50
n_steps_frac_high = 10
n_steps_frac_low = 10

predictor = StockPredictor(ticker,bartime,parentdir,frac_change_lower=frac_change_lower,frac_change_upper=frac_change_upper,
                           frac_high_upper=frac_high_upper,frac_low_upper=frac_low_upper,n_steps_frac_change=n_steps_frac_change,
                           n_steps_frac_high=n_steps_frac_high,n_steps_frac_low=n_steps_frac_low)
data = pd.read_csv('data/{p}/data/formatted/{t}/{t}_{b}_data_formatted.csv'.format(p=parentdir,t=ticker,b=bartime))
predictor.fit(data)
predictor.predict_close_prices_for_days(250,recent=True,with_plot=True,save_plot=True)
predictor.save_pred()
print(f'Predicted MAPE: {predictor.getMAPE()}')