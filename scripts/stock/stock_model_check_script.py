import pickle
from stock_trading import ModelChecker

def main():
    """
    Main function to execute the stock model check script.
    
    This function loads arrays, initializes a ModelChecker object, calculates the pnl for each time interval,
    and recompiles the losers for each time interval.
    """
    # load arrays
    sp500_tickers = pickle.load(open('stock/iterables/sp500_tickers.pkl', 'rb'))
    time_intervals = pickle.load(open('stock/iterables/time_intervals.pkl', 'rb'))
    
    model_checker = ModelChecker(ticker_list=sp500_tickers, 
                                 time_intervals=time_intervals, parentdir='stock', model_type='gaussian')
    # calculate the pnl for each time interval for all sp500 stocks
    pnl_df = model_checker.calc_all_pnl()
    
    # loop for each time interval and recompile the losers
    for bartime in time_intervals:    
        model_checker.recompile_losers(bartime, pnl_df)
    
if __name__=='__main__':
    main()