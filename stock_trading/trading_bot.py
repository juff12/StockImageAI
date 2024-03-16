from .stock_predictor import StockPredictor
from .coinbase_wrapper import CoinbaseWrapper
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time


class TradingBot(object):
    """
    A class representing a trading bot for stock trading.

    Parameters:
    - ticker (str): The ticker symbol of the stock.
    - bartime (str): The time interval for fetching stock data.
    - api_key (str): The API key for accessing stock data.
    - open_strategy (str): The opening strategy for placing trades.
    - close_strategy (str): The closing strategy for closing trades.
    - allow_shorts (bool): Whether short selling is allowed.
    - _type (str): The type of trading (e.g., stock, cryptocurrency).
    - model_type (str): The type of model used for predicting stock prices.
    """

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
        """
        Initialize the logger for logging trading bot activities.
        """
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)
    
    def get_timespan(self, step_size, step_unit, interval_size):
        """
        Calculate the timespan for a given step size, step unit, and interval size.

        Parameters:
        - step_size (int): The size of each step.
        - step_unit (str): The unit of each step (e.g., day, hour, minute).
        - interval_size (int): The size of the interval.

        Returns:
        - timespan (int): The calculated timespan.
        """
        return (step_size * interval_size if step_unit==name else 0 for name in ['day','hour','minute'])
    
    def get_current_price(self):
        """
        Get the current price of the stock.

        Returns:
        - current_price (float): The current price of the stock.
        """
        # format the ticker
        from_, to_ = self.ticker.upper().split('USD')[0], 'USD'
        # get the last price
        last_trade = self.client.get_last_crypto_trade(from_=self._ticker)
        return last_trade.price

    def get_data_for_interval(self, step_size, step_unit, interval_size=11):
        """
        Get the stock data for a given time interval.

        Parameters:
        - step_size (int): The size of each step.
        - step_unit (str): The unit of each step (e.g., day, hour, minute).
        - interval_size (int): The size of the interval.

        Returns:
        - interval_data (pd.DataFrame): The stock data for the interval.
        """
        # get days, hours, minutes of the time interval ('5_minutes', '15_minutes', '1_hour', '4_hours', '1_day')
        days, hours, minutes = self.get_timespan(step_size, step_unit, interval_size)
        # get all the information within the time interval for the time interval ('5_minutes', '15_minutes', '1_hour', '4_hours', '1_day')
        start = datetime.now() - timedelta(days=self.days, hours=self.hours, minutes=self.minutes)
        end = datetime.now()
        # get the data for the time interval
        interval_data = [a for a in self.client.list_aggs(ticker=self._ticker, multiplier=step_size,
                                                          timespan=step_unit,from_=start, to=end)]
        interval_data = pd.DataFrame(interval_data)
        return interval_data
    
    def get_minute_open(self):
        """
        Get the opening price for the current minute.

        Returns:
        - minute_open (float): The opening price for the current minute.
        """
        return

    def get_minute_close(self):
        """
        Get the closing price for the current minute.

        Returns:
        - minute_close (float): The closing price for the current minute.
        """
        min_close = self.client.get_snapshot_ticker(ticker=self._ticker).min.close
        return min_close

    def predict_price(self, open_price=None, step_size=None, step_unit=None):
        """
        Predict the price for the next time interval.

        Parameters:
        - open_price (float): The opening price for the current interval.
        - step_size (int): The size of each step.
        - step_unit (str): The unit of each step (e.g., day, hour, minute).

        Returns:
        - predicted_price (float): The predicted price for the next time interval.
        """
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
    
    def check_time(self, step_size, step_unit):
        """
        Check if the current time is a valid time for placing a trade.

        Parameters:
        - step_size (int): The size of each step.
        - step_unit (str): The unit of each step (e.g., day, hour, minute).

        Returns:
        - is_valid_time (bool): True if the current time is a valid time for placing a trade, False otherwise.
        """
        # get the current time and check that it is the end of interval (True) or not (False
        time = datetime.now()
        time_dict = {'day': time.day, 'hour': time.hour, 'minute': time.minute}
        interval = time_dict[step_unit]
        if interval % step_size == 0:
            return True
        return False
    
    def get_step_price(self, step_size, step_unit):
        """
        Get the predicted price for the next time interval based on the step size and step unit.

        Parameters:
        - step_size (int): The size of each step.
        - step_unit (str): The unit of each step (e.g., day, hour, minute).

        Returns:
        - predicted_price (float): The predicted price for the next time interval.
        """
        while True:
            if self.check_time(step_size, step_unit):
                break
        return self.predict_price(step_size, step_unit)

    def close_trade(self):
        """
        Close the current trade.

        Returns:
        - is_trade_closed (bool): True if the trade is closed, False otherwise.
        """
        # place the limits orders, update acording to strategy
        # update until trade closes
        # emergency market close
        
        # get updates from API to see that the trade is closed
        
        # return true if trade is closed
        return True

    def open_trade(self):
        """
        Open a new trade.

        Returns:
        - trade_strategy (tuple): The opening trade strategy and the predicted closing price.
        """
        # get the predicted price
        open_price = self.get_minute_open()
        pred_price = self.predict_price(open_price)
        if self.allow_shorts is False and open_price > pred_price:
            # dont take a trade
            return None
        return self.open_strategy(open_price, pred_price)
        
    def normal_open(self, open_price, pred_price, tol=0.005):
        """
        Define the opening trade strategy for normal trading.

        Parameters:
        - open_price (float): The opening price for the current interval.
        - pred_price (float): The predicted closing price for the current interval.
        - tol (float): The tolerance for placing the trade.

        Returns:
        - trade_strategy (tuple): The opening trade strategy and the predicted closing price.
        """
        # return a opening trade strategy and the predicted closing price
        current_price = self.get_current_price()
        if open_price < pred_price:
            return ([(current_price + current_price * tol, 1.0)], pred_price)
        return ([(current_price - current_price * tol, 1.0)], pred_price)

    def layered_open(self, open_price, pred_price, num=5):
        """
        Define the opening trade strategy for layered trading.

        Parameters:
        - open_price (float): The opening price for the current interval.
        - pred_price (float): The predicted closing price for the current interval.
        - num (int): The number of layers for placing the trade.

        Returns:
        - trade_strategy (tuple): The opening trade strategy and the predicted closing price.
        """
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
        """
        Define the opening trade strategy for stocastic trading.

        Parameters:
        - open_price (float): The opening price for the current interval.
        - pred_price (float): The predicted closing price for the current interval.
        - temp (int): The temperature for stocastic trading.
        - tol (float): The tolerance for placing the trade.

        Returns:
        - trade_strategy (tuple): The opening trade strategy and the predicted closing price.
        """
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
        """
        Define the opening trade strategy for stocastic layered trading.

        Parameters:
        - open_price (float): The opening price for the current interval.
        - pred_price (float): The predicted closing price for the current interval.

        Returns:
        - trade_strategy (tuple): The opening trade strategy and the predicted closing price.
        """
        return None

    def hmm_strategy(self):
        """
        Define the closing trade strategy using the Hidden Markov Model (HMM).

        Returns:
        - trade_strategy (tuple): The closing trade strategy and the predicted closing price.
        """
        return None
    
    def hill_climbing(self):
        """
        Define the closing trade strategy using the Hill Climbing algorithm.

        Returns:
        - trade_strategy (tuple): The closing trade strategy and the predicted closing price.
        """
        return None

    def simulated_annealing(self):
        """
        Define the closing trade strategy using the Simulated Annealing algorithm.

        Returns:
        - trade_strategy (tuple): The closing trade strategy and the predicted closing price.
        """
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
        """
        Get the local minimum and maximum prices for the stock.

        Returns:
        - local_minmax (float): The local minimum and maximum prices for the stock.
        """
        return None