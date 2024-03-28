import numpy as np
import pandas as pd

class TradingTechnicals(object):
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def moving_average(self, window: int, column: str):
        """
        Function to calculate the moving average of a column in the DataFrame.
        
        Parameters:
        window (int): The window size for the moving average.
        column (str): The column in the DataFrame to calculate the moving average.
        
        Returns:
        pd.Series: The moving average of the column.
        """
        return self.data[column].rolling(window=window).mean()
    
    def exponential_moving_average(self, window: int, column: str):
        """
        Function to calculate the exponential moving average of a column in the DataFrame.
        
        Parameters:
        window (int): The window size for the exponential moving average.
        column (str): The column in the DataFrame to calculate the exponential moving average.
        
        Returns:
        pd.Series: The exponential moving average of the column.
        """
        return self.data[column].ewm(span=window, adjust=False).mean()
    
    def relative_strength_index(self, window: int, column: str):
        """
        Function to calculate the relative strength index (RSI) of a column in the DataFrame.
        
        Parameters:
        window (int): The window size for the RSI calculation.
        column (str): The column in the DataFrame to calculate the RSI.
        
        Returns:
        pd.Series: The RSI of the column.
        """
        delta = self.data[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def keltners_channel(self, window, atr_window=20, column='close', upper_mult=2, lower_mult=2):
        """
        Function to calculate the Keltner's Channel of a column in the DataFrame.
        
        Parameters:
        window (int): The window size for the Keltner's Channel calculation.
        column (str): The column in the DataFrame to calculate the Keltner's Channel.
        
        Returns:
        pd.Series: The upper and lower bands of the Keltner's Channel.
        """
        exp = self.data[column].ewm(span=window, adjust=False).mean()
        atr = self.average_true_range(window=atr_window)
        upper_band = exp + upper_mult * atr
        lower_band = exp - lower_mult * atr
        return upper_band, exp, lower_band
    
    def average_true_range(self, window=20):
        """
        Function to calculate the Average True Range (ATR) of a column in the DataFrame.
        
        Parameters:
        window (int): The window size for the ATR calculation.
        
        Returns:
        pd.Series: The ATR of the column.
        """
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.ewm(span=window, adjust=False).mean()

    def bollinger_bands(self, window: int, column='close', mult_upper=2, mult_lower=2):
        """
        Function to calculate the Bollinger Bands of a column in the DataFrame.
        
        Parameters:
        window (int): The window size for the Bollinger Bands calculation.
        column (str): The column in the DataFrame to calculate the Bollinger Bands.
        
        Returns:
        pd.Series: The upper and lower bands of the Bollinger Bands.
        """
        sma = self.data[column].rolling(window=window).mean()
        std = self.data[column].rolling(window=window).std()
        upper_band = sma + mult_upper * std
        lower_band = sma - mult_lower * std
        return upper_band, sma, lower_band
    
    def volume_weighted_average_price(self, window=1, std_window=2, mult_upper=1.25, mult_lower=1.25):
        """
        Function to calculate the Volume Weighted Average Price (VWAP) of the DataFrame.
        
        Parameters:
        window (int): The window size for the VWAP calculation.
        
        Returns:
        pd.Series: The VWAP of the DataFrame.
        """
        hlc3 = (self.data['volume'] * (self.data['high'] + self.data['low'] + self.data['close']) / 3)
        vwap = hlc3.rolling(window=window).sum() / self.data['volume'].rolling(window=window).sum()
        print(vwap)
        std = vwap.rolling(window=std_window).std()
        upper_band = vwap + mult_upper * std
        lower_band = vwap - mult_lower * std
        return upper_band, vwap, lower_band
