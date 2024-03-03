from hmm_training import StockPredictor

stockpred = StockPredictor(('zts', '1_day'), multimodal=False)
stockpred.fit()
mape = stockpred.predict_close_prices_for_days(252, with_plot=True)
print(mape)