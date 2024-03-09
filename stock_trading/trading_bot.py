from .stock_predictor import StockPredictor
from .coinbase_wrapper import CoinbaseWrapper
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time


class TradingBot(object):
    def __init__(self, ticker, bartime, api_key,
                 open_strategy='normal', close_strategy='normal',
                 allow_shorts=False, _type='stock', model_type='gaussian'):
        self._init_logger()

        open_strats = {'normal': self.normal_open(), 'layered': self.layered_open(),
                       'stocastic': self.stocastic_open(), 'stocastic_layered': self.stocastic_layered_open()}
        close_strats = {'normal': self.hmm_strategy(), 'simulated_annealing': self.simulated_annealing(),
                        'hill_climbing': self.hill_climbing()}

        # set the closing and opening strategy function
        self.open_strategy = open_strats[open_strategy]
        self.close_strategy = close_strats[close_strategy]
        
        self.allow_shorts=allow_shorts

        # var for loading model
        self.ticker = ticker
        self.bartime = bartime
        
        # the predicted price for this time interval
        self.interval_pred = None
        
        self.predictor = StockPredictor(ticker, bartime, parentdir=_type,
                                        load_model=True, model_type=model_type)
        
        self.coinbase = CoinbaseWrapper(ticker)

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
        
        return

    def get_minute_close(self):
        min_close = self.client.get_snapshot_ticker(ticker=self._ticker).min.close
        return min_close

    def predict_price(self, open_price=None, step_size=None, step_unit=None):
        if open_price is None:
            open_price = self.get_minute_open()
        # detect unitialized steps
        if step_size is None or step_unit is None:
            step_size = self._multiplier
            step_unit = self._timespan
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

    def close_trade(self):
        
        
        # place the limits orders, update acording to strategy
        # update until trade closes
        # emergency market close
        
        # get updates from API to see that the trade is closed
        
        # return true if trade is closed
        return True

    def open_trade(self):
        # get the predicted price
        open_price = self.get_minute_open()
        pred_price = self.predict_price(open_price)
        if self.allow_shorts is False and open_price > pred_price:
            # dont take a trade
            return None
        return self.open_strategy(open_price, pred_price)
        
    def normal_open(self, open_price, pred_price, tol=0.005):
        # return a opening trade strategy and the predicted closing price
        current_price = self.get_current_price()
        if open_price < pred_price:
            return ([(current_price + current_price * tol, 1.0)], pred_price)
        return ([(current_price - current_price * tol, 1.0)], pred_price)

    def layered_open(self, open_price, pred_price, num=5):
        local_minmax = self.getLocalMinMax()
        if local_minmax < open_price:
            # creates a layered order of five orders split evenly between 
            # the recent low and current price
            layered_order = list(np.linspace(local_minmax, open_price, num))
            return ([(order_price, 1/num) for order_price in layered_order], pred_price)
        else:
            # creates a layered order of five orders split evenly between 
            # the recent low and current price
            layered_order = list(np.linspace(open_price, local_minmax, num))
            return ([(order_price, 1/num) for order_price in layered_order], pred_price)
    
    def stocastic_open(self, open_price, pred_price, temp=10, tol=0.005):
        if open_price < pred_price:
            # longing
            prev_price = self.get_current_price()
            while True:
                if np.random.randint(0,temp) == 0:
                    return ([(prev_price - prev_price * tol, 1.0)], pred_price)
                
                current_price = self.get_current_price()
                
                if current_price < prev_price:
                    temp -= 1
                    prev_price = current_price
                time.sleep()
        else:
            # shorting
            return None

    def stocastic_layered_open(self, open_price, pred_price,):
        return None

    def hmm_strategy(self):
        return None
    
    def hill_climbing(self):
        return None

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
    
    def getLocalMinMax(self):
        return None