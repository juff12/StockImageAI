import pickle
from stock_trading import ModelChecker
import pandas as pd
def main():
    # load arrays
    coinbase_tickers = pickle.load(open('data/crypto/iterables/coinbase_tickers.pkl', 'rb'))
    time_intervals = pickle.load(open('data/crypto/iterables/time_intervals.pkl', 'rb'))
    
    model_checker = ModelChecker(ticker_list=coinbase_tickers, 
                                 time_intervals=time_intervals, parentdir='crypto', model_type='gaussian')
    
    model_checker.calc_all_pnl(save_steps=True)
    # for bartime in time_intervals:    
    #     model_checker.recompile_losers(bartime, pnl_df)
    
if __name__=='__main__':
    main()