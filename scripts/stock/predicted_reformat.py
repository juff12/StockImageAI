import pandas as pd
from pathlib import Path
import pickle

def reformat_pred(df_pred, df_orig):
    """
    Reformat the predicted values based on the original dataframe.

    Args:
        df_pred (pandas.DataFrame): DataFrame containing the predicted values.
        df_orig (pandas.DataFrame): DataFrame containing the original values.

    Returns:
        pandas.DataFrame: Reformatted DataFrame with the predicted values aligned with the original values.
    """
    # get the date range of the predicted values
    df_pred.reset_index(inplace=True)
    dates = df_pred['date'].values # gets the dates to format
    df_orig.set_index('date', inplace=True) # allows for date indexing
    df_out = df_orig[['open','high','low','close','volume']].loc[dates]
    # if the format for the column labels is wrong, fix it
    if 'pred close' in df_pred:
       df_out.insert(4, 'pred', df_pred['pred close'].values)
    else:
        df_out.insert(4, 'pred', df_pred['predicted'].values)
    
    return df_out

def main():
    """
    Main function to reformat the predicted data for each stock and time interval.

    Reads in the iterables for S&P 500 tickers, NASDAQ 100 tickers, and time intervals.
    Adds the high, low, and open values to the predicted datasets for each stock and time interval.
    Reformats the predicted data by inserting the predicted values into the original dataframe.
    Saves the reformatted predicted data to a CSV file.

    Returns:
        None
    """
    # read in the iterables
    sp500_tickers = pickle.load(open('data/stock/iterables/sp500_tickers.pkl', 'rb'))
    nasdaq_100_tickers = pickle.load(open('data/stock/iterables/nasdaq_100_tickers.pkl', 'rb'))
    time_intervals = pickle.load(open('data/stock/iterables/time_intervals.pkl', 'rb'))

    # add the high, low, open to the predicted datasets for each stock and time interval
    for ticker in sp500_tickers:
        for bartime in time_intervals:
            # load the predicted and original data
            df_pred = pd.read_csv('data/stock/predicted/{t}/{t}_{b}_pred.csv'.format(t=ticker,b=bartime))
            df_orig = pd.read_csv('data/stock/data/formatted/{t}/{t}_{b}_data_formatted.csv'.format(t=ticker,b=bartime))
            # reformat the predicted data by inserting the predicted values into the original dataframe
            df_pred = reformat_pred(df_pred,df_orig)
            filepath = Path('data/stock/predicted/{t}/{t}_{b}_pred.csv'.format(t=ticker,b=bartime))
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df_pred.to_csv(filepath)

if __name__ == '__main__':
    main()