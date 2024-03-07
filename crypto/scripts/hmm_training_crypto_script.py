import pickle
import os
from StockPredictor import StockPredictor
import pandas as pd
from pathlib import Path

def main():
    # read in the iterables
    coinbase_tickers = pickle.load(open('crypto/iterables/coinbase_tickers.pkl', 'rb'))
    time_intervals = pickle.load(open('crypto/iterables/time_intervals.pkl', 'rb'))
    
    mapes = []
    
    # crate model for each stock in S&P500
    for ticker in coinbase_tickers:
        
        ticker_mape = [ticker]
        
        for bartime in time_intervals:
            stock_predictor = StockPredictor(ticker=ticker, bartime=bartime, 
                                             parentdir='crypto', load_model=False, model_type='gaussian')
            data = pd.read_csv('crypto/data/formatted/{t}/{t}_{b}_data_formatted.csv'.format(t=ticker,b=bartime))
            stock_predictor.fit(data,test_size=0.33)
            stock_predictor.save_model()
            
            # set to 252 to predict 1 year of trading data for 1_day
            stock_predictor.predict_close_prices_for_days(252, with_plot=True, save_plot=True)
            stock_predictor.save_pred()
            
            ticker_mape.append(stock_predictor.getMAPE())
        
        mapes.append(ticker_mape) # add all mapes for stock
        
        # periodically save the mape in case of crash when iterating over data
        filepath = Path('crypto/data/mapes/mapes_temp.pkl')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as file: 
            pickle.dump(mapes, file)
    
    # save mapes
    df_mape = pd.DataFrame(mapes, columns=['ticker','1_day','4_hour','1_hour'])
    filepath = Path('crypto/data/mapes/crypto_mapes.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_mape.to_csv(filepath,index=False)

if __name__=='__main__':
    main()