import numpy as np
import pandas as pd
from pathlib import Path
import pickle

def reformat_pred(df_pred, df_orig):
    # get the date range of the predicted values
    df_pred.reset_index(inplace=True)
    dates = df_pred['date'].values # gets the dates to format
    df_orig.set_index('date', inplace=True) # allows for date indexing
    df_out = df_orig[['open','high','low','close','volume']].loc[dates]
    if 'pred close' in df_pred:
       df_out.insert(4, 'pred', df_pred['pred close'].values)
    else:
        df_out.insert(4, 'pred', df_pred['predicted'].values)
    
    return df_out

def main():
    # read in the iterables
    sp500_tickers = pickle.load(open('iterables/sp500_tickers.pkl', 'rb'))
    nasdaq_100_tickers = pickle.load(open('iterables/nasdaq_100_tickers.pkl', 'rb'))
    time_intervals = pickle.load(open('iterables/time_intervals.pkl', 'rb'))

    # add the high, low, open to the predicted datasets
    for ticker in sp500_tickers:
        for bartime in time_intervals:
            df_pred = pd.read_csv('stock/data/predicted/{t}/{t}_{b}_pred.csv'.format(t=ticker,b=bartime))
            df_orig = pd.read_csv('stock/data/formatted/{t}/{t}_{b}_data_formatted.csv'.format(t=ticker,b=bartime))
            df_pred = reformat_pred(df_pred,df_orig)
            filepath = Path('stock/data/predicted/{t}/{t}_{b}_pred.csv'.format(t=ticker,b=bartime))
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df_pred.to_csv(filepath)

if __name__ == '__main__':
    main()