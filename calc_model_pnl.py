import pickle
import pandas as pd

def main():
    sp500_tickers = pickle.load(open('iterables/sp500_tickers.pkl', 'rb'))
    nasdaq_100_tickers = pickle.load(open('iterables/nasdaq_100_tickers.pkl', 'rb'))
    time_intervals = pickle.load(open('iterables/time_intervals.pkl', 'rb'))

    idx_time = {'1_day': 0, '4_hour': 1, '1_hour': 2}
    
    # df = pd.read_csv(f'data/predicted/aapl/aapl_1_day_pred.csv')
    # diff = []
    # for idx, row in df.iterrows():
    #     act_diff = row['close'] - row['open']
    #     pred_diff = row['pred'] - row['open']

    #     if ((pred_diff >= 0 and act_diff >= 0) or (pred_diff <= 0 and act_diff <= 0)):
    #         diff.append(abs(act_diff))
    #     else:
    #         diff.append(-1*abs(act_diff))
    # print(diff)
    # print(sum(diff))
    pnl_dict = {}
    for ticker in sp500_tickers:
        pnl_time = [0,0,0]
        for bartime in time_intervals:
            data = pd.read_csv('data/predicted/{t}/{t}_{b}_pred.csv'.format(t=ticker,b=bartime))
            for idx, row in data.iterrows():
                pred_diff = row['open'] - row['pred']
                actual_diff = row['open'] - row['close']
                # if trading in same direction 
                if ((pred_diff >= 0 and actual_diff >= 0) or 
                    (pred_diff <= 0 and actual_diff <= 0)):
                    pnl_time[idx_time[bartime]] += abs(actual_diff)
                else: # assumes trade at open 
                    pnl_time[idx_time[bartime]] -= abs(actual_diff)
        pnl_dict[ticker] = pnl_time
    df = pd.DataFrame.from_dict(pnl_dict, orient='index', columns=['1 day', '4 hour', '1 hour'])
    # df.to_csv('pnl_summary.csv')
if __name__=='__main__':
    main()