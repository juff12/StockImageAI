import requests
from datetime import datetime
import pandas as pd
import logging

class CoinbaseWrapper(object):
    def __init__(self, ticker: str):
        self._init_logger()
        self.bartimes_convert = {'1_min': 60, '5_minute': 300, '15_minute': 900, 
                                 '1_hour': 3600, '6_hour': 21600, '1_day': 86400}
        self.ticker = ticker.upper().replace('USD','-USD')

    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

    def fetch_last_n_candles(self, bartime, limit):
        # helper for 4_hour
        if bartime == '4_hour':
            candles_df = self.fetch_last_n_candles('1_hour', 40)
            if candles_df is not None:
                return self.convert_4hour_data(candles_df)
            return None # failed

        # get the request size
        bartime = self.bartimes_convert[bartime]
        
        url = 'https://api.pro.coinbase.com/products/{t}/candles?granularity={b}&limit={l}'.format(t=self.ticker,
                                                                                                   b=bartime,
                                                                                                   l=limit)
        response = requests.get(url)
        
        if response.status_code == 200:
            candles = response.json()
            if candles is not None:
                candles.reverse()
                candles_df = pd.DataFrame(candles, columns=['date','low','high','open','close','volume'])
                candles_df['date'] = candles_df['date'].apply(lambda x: datetime.utcfromtimestamp(x))
                return candles_df
        self._logger.error(f"Failed to fetch {self.ticker} candles data. Status code: {response.status_code}")
        return None
    
    def convert_4hour_data(self, candles_df: pd.DataFrame):
        print(candles_df)
        start, end = None, None
        four_hour_candles = []
        for idx, row in candles_df.iterrows():
            if len(four_hour_candles) >= 75:
                break # breaks out before it throws error for index not found
            if (row['date'].hour % 4 == 0) and end is None:
                end = idx+1
            elif (row['date'].hour % 4 == 0) and start is None:
                start = idx+1

                date = candles_df.at[end-1,'date']
                low = min(candles_df.iloc[end:start]['low'].values) # minimum low value in period
                high = max(candles_df.iloc[end:start]['high'].values) # maximum high value in period
                open = candles_df.at[start-1,'open'] # the opening price at start
                close = candles_df.at[end,'close'] # the closing price at end
                volume = sum(candles_df.iloc[end:start]['volume'].values) # the total volume
                four_hour_candles.append([date,low,high,open,close,volume])

                # reset index's
                start, end = None, start
        #four_hour_candles.reverse()
        return pd.DataFrame(four_hour_candles, columns=['date','low','high','open','close','volume'])

    def get_current_price(self):
        # Coinbase API endpoint for currency price
        url = f"https://api.coinbase.com/v2/prices/{self.ticker}/spot"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            # Extracting the current price of the given currency
            currency_price = data['data']['amount']
            
            return float(currency_price)
        
        except Exception as e:
            self._logger.error("Error occurred:", e)
            return None

# Example usage
if __name__ == "__main__":
    coinbase_wrapper = CoinbaseWrapper('btcusd')
    btcusd_candles = coinbase_wrapper.fetch_last_n_candles('4_hour', 10)
    print(btcusd_candles)