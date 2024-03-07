import pickle
from ModelChecker import ModelChecker

def main():
    # load arrays
    sp500_tickers = pickle.load(open('stock/iterables/sp500_tickers.pkl', 'rb'))
    time_intervals = pickle.load(open('stock/iterables/time_intervals.pkl', 'rb'))
    
    model_checker = ModelChecker(ticker_list=sp500_tickers, 
                               time_intervals=time_intervals, parentdir='stock', model_type='gaussian')
    pnl_df = model_checker.calc_all_pnl()
    for bartime in time_intervals:    
        model_checker.recompile_losers(bartime, pnl_df)
    
if __name__=='__main__':
    main()