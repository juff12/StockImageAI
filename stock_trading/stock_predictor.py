import pickle
import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM, GMMHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
from timeit import default_timer as timer
from datetime import date, datetime, timedelta

# Supress warning in hmmlearn
warnings.filterwarnings("ignore")
# Change plot style to ggplot (for better and more aesthetic visualisation)
plt.style.use('ggplot')

class StockPredictor(object):
    """
    A class for predicting stock prices using Hidden Markov Models (HMM).

    Args:
        ticker (str): The ticker symbol of the stock.
        bartime (str): The time interval of the stock data (e.g., '1d', '1h', '15m').
        parentdir (str, optional): The parent directory for saving/loading data and models. Defaults to 'stock'.
        load_model (bool, optional): Whether to load a pre-trained model. Defaults to False.
        model_type (str, optional): The type of model to use ('gaussian' or 'gmmhmm'). Defaults to 'gaussian'.
        n_hidden_states (int, optional): The number of hidden states in the HMM. Defaults to 4.
        n_latency_days (int, optional): The number of previous days to consider for prediction. Defaults to 10.
        frac_change_lower (float, optional): The lower bound of fraction change in prices. Defaults to -0.1.
        frac_change_upper (float, optional): The upper bound of fraction change in prices. Defaults to 0.1.
        frac_high_upper (float, optional): The upper bound of fraction change in high prices. Defaults to 0.1.
        frac_low_upper (float, optional): The upper bound of fraction change in low prices. Defaults to 0.1.
        n_steps_frac_change (int, optional): The number of steps for fraction change. Defaults to 50.
        n_steps_frac_high (int, optional): The number of steps for fraction change in high prices. Defaults to 10.
        n_steps_frac_low (int, optional): The number of steps for fraction change in low prices. Defaults to 10.
        n_mix (int, optional): The number of mixture components for GMMHMM. Defaults to 5.
    """
    def __init__(self, ticker, bartime, parentdir='stock', load_model=False, 
                 model_type='gaussian', n_hidden_states=4, n_latency_days=10,
                 frac_change_lower=-0.1, frac_change_upper=0.1, frac_high_upper=0.1,
                 frac_low_upper=0.1, n_steps_frac_change=50, n_steps_frac_high=10,
                 n_steps_frac_low=10, n_mix=5):
        self._init_logger()
        # set tqdm barformat
        self.bar_format ='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        self.ticker = ticker
        self.bartime = bartime
        self.parentdir = parentdir
        self.mape = None
        self.predictions = None
 
        self.n_latency_days = n_latency_days

        # set the model type name
        self.model_type = model_type

        if load_model:
            self.load_model()
        else:
            if self.model_type == 'gmmhmm':
                self.hmm = GMMHMM(n_components=n_hidden_states, n_mix=n_mix)
            else:
                self.hmm = GaussianHMM(n_components=n_hidden_states)
        
        self._compute_all_possible_outcomes(frac_change_lower, frac_change_upper, frac_high_upper,
                                            frac_low_upper, n_steps_frac_change, n_steps_frac_high, n_steps_frac_low)
    
    def _init_logger(self):
        """
        Initialize the logger for logging messages.
        """
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

    def load_test(self, test_data):
        """
        Load the test data for prediction.

        Args:
            test_data (pd.DataFrame): The test data as a pandas DataFrame.
        """
        self._logger.info('>>> Loading the test data')
        self._test_data=test_data
        self._logger.info('>>> Test data loaded')
        
    def load_model(self, filepath=None):
        """
        Load a pre-trained model.

        Args:
            filepath (str, optional): The filepath of the model. Defaults to None.
        """
        self._logger.info('>>> Loading the model')
        # load the model that the ticker was constructed on
        if filepath is None:
            filepath = Path('data/{p}/models/{m}/{t}/model_{t}_{b}.pkl'.format(p=self.parentdir,
                                                                               m=self.model_type,
                                                                               t=self.ticker,
                                                                               b=self.bartime))
        try:
            self.hmm = pickle.load(open(filepath, 'rb'))
            self._logger.info('>>> Model Loaded')
        except:
            self._logger.error('The model could not be loaded')

    def _split_train_test_data(self, data, test_size):
        """
        Splits the given data into training and testing sets.

        Parameters:
        - data: The input data to be split.
        - test_size: The proportion of the data to be used for testing.

        Returns:
        - None

        This method splits the input data into two sets: a training set and a testing set.
        The training set is used to train the model, while the testing set is used to evaluate its performance.
        The `test_size` parameter determines the proportion of the data to be used for testing.
        """
        _train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)

        self._train_data = _train_data
        self._test_data = test_data
 
    @staticmethod
    def _extract_features(data: pd.DataFrame):
        """
        Extracts features from the given data.

        Parameters:
        - data: A dataframe object containing 'open', 'close', 'high', and 'low' values.

        Returns:
        - features: A numpy array containing the computed fraction change in close, high, and low prices.
        """
        open = np.array(data['open'])
        close = np.array(data['close'])
        high = np.array(data['high'])
        low = np.array(data['low'])
    
        # Compute the fraction change in close, high and low prices
        # which would be used as features
        frac_change = (close - open) / open
        frac_high = (high - open) / open
        frac_low = (open - low) / open
    
        return np.column_stack((frac_change, frac_high, frac_low))

    def fit(self, data: pd.DataFrame, train_start_date=None, test=True, test_size=0.33):
            """
            Fits the stock predictor model to the given data.

            Parameters:
            - data (pd.DataFrame): The input data containing stock information.
            - train_start_date (datetime.date, optional): The start date for training the model. If not provided, the entire dataset will be used for training.
            - test (bool, optional): Whether to perform testing after training. Default is True.
            - test_size (float, optional): The proportion of data to be used for testing. Default is 0.33.

            Returns:
            None
            """
            # if a start date is provided, use the data from that date onwards
            if train_start_date != None:
                data['date'] = data['date'].astype('datetime64[ns]')
                train_start = data['date'].index[data['date'].dt.date == train_start_date.date()].to_list()[0]
                last_n_entries = len(data) - train_start
                data = data.tail(last_n_entries)
            # split the data into training and testing sets
            self._split_train_test_data(data, test_size)
            self._logger.info('>>> Extracting Features')
            feature_vector = StockPredictor._extract_features(self._train_data)
            self._logger.info('Features extraction Completed <<<')
            
            self.hmm.fit(feature_vector)
        
    def save_model(self, filepath=None):
        """
        Save the model to a file.

        Args:
            filepath (str or Path, optional): The path where the model should be saved.
                If not provided, a default path will be used based on the parent directory,
                model type, ticker, and bartime.

        Returns:
            None
        """
        # get appropriate file path if none given
        if filepath is None:
            filepath = Path("data/{p}/models/{m}/{t}/model_{t}_{b}.pkl".format(p=self.parentdir,
                                                                               m=self.model_type,
                                                                               t=self.ticker,
                                                                               b=self.bartime))
        filepath = Path(filepath)
        # Create the directory if it does not exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as file:
            pickle.dump(self.hmm, file)
    
    
    def _compute_all_possible_outcomes(self, frac_change_lower, frac_change_upper, frac_high_upper,
                                       frac_low_upper, n_steps_frac_change, n_steps_frac_high, n_steps_frac_low):
        """
        Compute all possible outcomes based on the given parameters.

        Args:
            frac_change_lower (float): The lower bound of the fractional change range.
            frac_change_upper (float): The upper bound of the fractional change range.
            frac_high_upper (float): The upper bound of the fractional high range.
            frac_low_upper (float): The upper bound of the fractional low range.
            n_steps_frac_change (int): The number of steps in the fractional change range.
            n_steps_frac_high (int): The number of steps in the fractional high range.
            n_steps_frac_low (int): The number of steps in the fractional low range.

        Returns:
            None

        """
        # compute a numpy array of the possible outcomes between the given fracChange ranges
        frac_change_range = np.linspace(frac_change_lower, frac_change_upper, n_steps_frac_change)
        # compute a numpy array of the possible outcomes between the given fracHigh ranges
        frac_high_range = np.linspace(0, frac_high_upper, n_steps_frac_high)
        # compute a numpy array of the possible outcomes between the given fracLow ranges
        frac_low_range = np.linspace(0, frac_low_upper, n_steps_frac_low)
        self._possible_outcomes = np.array(list(itertools.product(
            frac_change_range, frac_high_range, frac_low_range)))
    
    def _get_most_probable_outcome(self, day_index):
        """
        Get the most probable outcome for a given day index.

        Parameters:
        - day_index (int): The index of the day for which to calculate the most probable outcome.

        Returns:
        - most_probable_outcome: The most probable outcome for the given day index.
        """
        # get the index of start and end
        previous_data_start_index = max(0, day_index - self.n_latency_days)
        previous_data_end_index = max(0, day_index - 1)
        # get the previous data
        previous_data = self._test_data.iloc[previous_data_end_index: previous_data_start_index]
        # extract the features from the previous data
        previous_data_features = StockPredictor._extract_features(previous_data)

        # ammend each possible outcome to the previous data data and score them
        # by using the HMM method on each set of previous days + possible outcome
        outcome_score = []
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack((previous_data_features, possible_outcome))
            outcome_score.append(self.hmm.score(total_data))
        # choose the largest score
        most_probable_outcome = self._possible_outcomes[np.argmax(outcome_score)]

        return most_probable_outcome

    def predict_close_price(self, day_index):
        """
        Predicts the close price of a stock for a given day index.

        Parameters:
        - day_index (int): The index of the day for which to predict the close price.

        Returns:
        - float: The predicted close price of the stock.
        """
        open_price = self._test_data.iloc[day_index]['open']
        predicted_frac_change, _, _ = self._get_most_probable_outcome(day_index)
        
        # return the predicted close price
        return open_price * (1 + predicted_frac_change)

    def predict_close_prices_for_days(self, days, recent=False, with_plot=False, save_plot=False):
        """
        Predicts the close prices for the given number of days.

        Args:
            days (int): The number of days to predict close prices for.
            recent (bool, optional): If True, uses the most recent data for prediction. Defaults to False.
            with_plot (bool, optional): If True, generates a plot of the predicted price and actual close prices. Defaults to False.
            save_plot (bool, optional): If True, saves the plot as an image file. Defaults to False.

        Returns:
            None

        Raises:
            None

        """
        # check if test data has been loaded
        if self._test_data is None:
            self._logger.error('No test data has been loaded')
            return None
        
        temp_test = self._test_data
        # if the given days are more than the remaining test data, use all test data
        if days > len(self._test_data):
            days = len(self._test_data)
        elif recent: # set the test data to the most recent data
            self._test_data = self._test_data[len(self._test_data)-days:]
        
        self._logger.info('>>> Beginning Prediction Generation')
        # generate the predicted close prices for the given number of days
        predicted_close_prices = []
        for day_index in tqdm(range(days), bar_format=self.bar_format, desc='Predictions'):
            predicted_close_prices.append(self.predict_close_price(day_index))
        self._logger.info('Prediction Generation Complete <<<')
        
        # get the actual close prices and predicted
        test_data = self._test_data[0: days]
        actual_close_prices = test_data['close']
        
        # save the actual and predicted close prices
        self.predictions = predicted_close_prices
        self.actual_close_data = test_data
        
        # save the MAPE
        self.mape = self.compute_mape(days, predicted_close_prices, actual_close_prices.values)
        
        # generate a plot of the predicted price and actual close prices
        if with_plot:
            days = np.array(test_data['date'], dtype="datetime64[ms]")

            fig = plt.figure(figsize=(24,12))
            
            axes = fig.add_subplot(111)
            axes.plot(days, actual_close_prices, 'bo-', markersize=3, label="actual")
            axes.plot(days, predicted_close_prices, 'ro-',markersize=3, label="predicted")
            axes.set_title(f'{self.ticker.upper()} - Predicted vs Actual Prices ({self.bartime})')
            axes.set_xlabel('Date')
            axes.set_ylabel('Value (US Dollars)')
            fig.autofmt_xdate()

            plt.legend()
            # save the plot if requested to save
            if save_plot:
                filepath = Path('data/{p}/charts/{m}_charts/{t}/{t}_{b}_chart.png'.format(p=self.parentdir,
                                                                                          m=self.model_type,
                                                                                          t=self.ticker,
                                                                                          b=self.bartime))
                filepath.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(filepath)
            else: # show the plot if not saved
                plt.show()
        
        # change test data back
        self._test_data = temp_test
        
        
        
    def compute_mape(self, days, predicted_close_prices, actual_close_prices):
        """
        Computes the Mean Absolute Percentage Error (MAPE) for a given number of days.

        Parameters:
        - days (int): The number of days to compute MAPE for.
        - predicted_close_prices (list): A list of predicted closing prices.
        - actual_close_prices (list): A list of actual closing prices.

        Returns:
        - mape (float): The computed MAPE value rounded to 2 decimal places.
        """
        # if days is greater than the length of the test data, set days to the length of the test data
        if days > len(self._test_data):
            days = len(self._test_data)
        
        mape = []
        # calculate the MAPE for each day
        for day in range(0,days):
            p, y = predicted_close_prices[day], actual_close_prices[day]
            diff_percent = (abs(p - y) / abs(y)) * 100
            mape.append(diff_percent)
        # calculate the average MAPE
        mape = sum(mape) / days
        
        return round(mape, 2)
    
    def save_pred(self, filepath=None):
        """
        Save the predicted stock data to a CSV file.

        Args:
            filepath (str or Path, optional): The file path to save the CSV file. If not provided,
                a default file path will be used based on the parent directory, model type, ticker,
                and bar time.

        Returns:
            None
        """
        df = self.actual_close_data[['date','open','high','low','close']]
        # round the predictions to 2 decimal places
        df.loc[:,'pred'] = [round(pred, 2) for pred in self.predictions]
        # create appropriate file path
        if filepath is None:
            filepath = Path("data/{p}/predicted/{m}/{t}/{t}_{b}_pred.csv".format(p=self.parentdir,
                                                                                 m=self.model_type,
                                                                                 t=self.ticker,
                                                                                 b=self.bartime))
        filepath = Path(filepath)
        # Create the directory if it does not exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)

    def predict_next_timeframe_price(self, data: pd.DataFrame):
        """
        Predicts the price for the next timeframe based on the given data.

        Args:
            data (pd.DataFrame): The input data used for prediction.

        Returns:
            float: The predicted price for the next timeframe.
        """
        self._test_data = data
        return self.predict_close_price(len(self._test_data) - 1)
    
    def getPredictions(self):
        """
        Returns a DataFrame containing the date, open, high, low, close, and predicted values.

        Returns:
            pandas.DataFrame: DataFrame with columns 'date', 'open', 'high', 'low', 'close', and 'pred'.
        """
        df = self.actual_close_data[['date','open','high','low','close']]
        df.loc[:,'pred'] = [round(pred, 2) for pred in self.predictions]
        return df

    def getMAPE(self):
        """
        Returns the Mean Absolute Percentage Error (MAPE) of the stock predictor.

        The MAPE is a measure of the accuracy of the stock predictor's forecasts.
        It calculates the average percentage difference between the predicted and actual values.

        Returns:
            float: The MAPE of the stock predictor.
        """
        return self.mape
