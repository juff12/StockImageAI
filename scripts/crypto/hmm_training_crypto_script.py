import pickle
from stock_trading import StockPredictor
import pandas as pd
from pathlib import Path
from datetime import datetime

def main():
    parentdir = 'crypto'

    # read in the iterables
    coinbase_tickers = pickle.load(open(f'data/{parentdir}/iterables/coinbase_tickers.pkl', 'rb'))
    time_intervals = pickle.load(open(f'data/{parentdir}/iterables/time_intervals.pkl', 'rb'))
    
    mapes = []
    
    bartime_presets = {'1_day': (datetime(2022,1,1,3,0,0)), '4_hour': (datetime(2023,3,1,3,0,0)), '1_hour': (datetime(2023,3,1,0,0,0)), '15_minute': (datetime(2023,3,1,3,0,0)), '5_minute': (datetime(2023,3,1,3,0,0))}
    
    # crate model for each crypto on coinbase
    for ticker in coinbase_tickers:
        
        ticker_mape = [ticker]
        
        for bartime in time_intervals:
            stock_predictor = StockPredictor(ticker=ticker.lower(), bartime=bartime, 
                                             parentdir=parentdir, load_model=False, model_type='gaussian')
            data = pd.read_csv('data/{p}/data/formatted/{t}/{t}_{b}_data_formatted.csv'.format(p=parentdir,
                                                                                               t=ticker,b=bartime))
            stock_predictor.fit(data,train_start_date=datetime(2023,3,1,3,0,0),test_size=0.2)
            stock_predictor.save_model()
            
            # set to 252 to predict 1 year of trading data for 1_day
            stock_predictor.predict_close_prices_for_days(252, recent=True, with_plot=True, save_plot=True)
            stock_predictor.save_pred()
            
            ticker_mape.append(stock_predictor.getMAPE())
        
        mapes.append(ticker_mape) # add all mapes for stock
        
        # periodically save the mape in case of crash when iterating over data
        filepath = Path(f'data/{parentdir}/mapes/mapes_temp.pkl')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as file: 
            pickle.dump(mapes, file)
    
    # save mapes
    col_labels = [ticker] + time_intervals
    
    df_mape = pd.DataFrame(mapes, columns=col_labels)
    filepath = Path(f'data/{parentdir}/mapes/{parentdir}_mapes.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_mape.to_csv(filepath,index=False)

if __name__=='__main__':
    main()