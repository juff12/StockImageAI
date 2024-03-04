import os
from pathlib import Path
from tqdm import tqdm
import pickle


def main():
    # read in the arrays for tickers
    sp500_tickers = pickle.load(open('iterables/sp500_tickers.pkl', 'rb'))
    nasdaq_100_tickers = pickle.load(open('iterables/nasdaq_100_tickers.pkl', 'rb'))
    
    python_path = str(Path.cwd()).replace('repos\StockImageAI','') + '\\anaconda3\python.exe '
    script_path = str(Path.cwd()) + '\chart_image_create.py '
    for ticker in tqdm(sp500_tickers, desc='Stocks: '):
        ticker = ticker.lower()
        path = python_path + script_path + ticker
        os.system(path)

if __name__ == '__main__':
    main()