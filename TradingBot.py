from StockPredictor import StockPredictor
from polygon import RESTClient
import logging
from datetime import datetime, timedelta
import pandas as pd

class TradingBot(object):
    def __init__(self, ticker, bartime, api_key, 
                 _type='stock', model_type='gaussian'):
        self._init_logger()

        # var for loading model
        self.ticker = ticker
        self.bartime = bartime

        # var for api
        self._multiplier, self._timespan = bartime.split('_')
        #convert multiplier to int
        self._multiplier = int(self._multiplier)

        if self._timespan == 'min': # converts for RESTClient
            self._timespan = 'minute'
        self._ticker = ticker
        if self._ticker == 'crypto': # convert for crypto
            self._ticker = 'X:' + self._ticker
        
        # the predicted price for this time interval
        self.interval_pred = None
        
        self.predictor = StockPredictor(ticker, bartime, parentdir=_type,
                                        load_model=True, model_type=model_type)
        self.client = RESTClient(api_key=api_key)

    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)
    
    def get_timespan(self, step_size, step_unit, interval_size):
        return (step_size * interval_size if step_unit==name else 0 for name in ['day','hour','minute'])
    
    def get_current_price(self):
        from_, to_ = self.ticker.upper().split('USD')[0], 'USD'
        last_trade = self.client.get_last_crypto_trade(from_=self._ticker)
        return last_trade.price

    def get_data_for_interval(self, step_size, step_unit, interval_size=11):
        days, hours, minutes = self.get_timespan(step_size, step_unit, interval_size)
        start = datetime.now() - timedelta(days=self.days, hours=self.hours, minutes=self.minutes)
        end = datetime.now()
        interval_data = [a for a in self.client.list_aggs(ticker=self._ticker, multiplier=step_size,
                                                          timespan=step_unit,from_=start, to=end)]
        interval_data = pd.DataFrame(interval_data)
        return interval_data
    
    def get_minute_open(self):
        min_open = self.client.get_snapshot_ticker(ticker=self._ticker).min.open
        return min_open

    def get_minute_close(self):
        min_close = self.client.get_snapshot_ticker(ticker=self._ticker).min.close
        return min_close

    def predict_price(self, step_size, step_unit):
        open_price = self.get_minute_open()

        # data for the time interval, default is 11 (10 periods)
        interval_data = self.get_data_for_interval(step_size, step_unit)
        interval_data.loc[len(interval_data)] = [open_price] + [0 for _ in range(0, len(interval_data.iloc[0])-1)]

        # predict the close on the given data
        return self.predictor.predict_next_timeframe_price(interval_data)
    
    def check_time(self, step_size, step_unit): # check if this is fast enough
        time = datetime.now()
        time_dict = {'day': time.day, 'hour': time.hour, 'minute': time.minute}
        interval = time_dict[step_unit]
        if interval % step_size == 0:
            return True
        return False
    
    def get_step_price(self, step_size, step_unit):
        while True:
            if self.check_time(step_size, step_unit):
                break
        return self.predict_price(step_size, step_unit)

    def simulated_annealing(self):
        # this function is called at the start of the inverval, grab the prediction
        self.interval_pred = self.predict_price(self._multiplier, self._timespan)

        start_time = datetime.now()
        days, hours, minutes = self.get_timespan(5, 'minutes', 1)
        end_time = datetime.now() - timedelta(days=self.days, hours=self.hours, minutes=self.minutes)
        
        while start_time < end_time:
            x=0

        # use simualted annealing to close a trade
        # close we are to the end of the time period,
        # while in profit, more likely to close trade

        # the probability of closing a trade increases as we approach the predicted price
        # and as we approach the end of the time interval
        # if we go beyond the predicted price, we increase the probability of closing trade

        return