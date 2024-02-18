import pandas as pd
from lightweight_charts import Chart
from pathlib import Path
import sys
import time

# interval of viewing the chart
interval = {'day': 120, '4_hour': 120, '1_hour': 120}
# amount to increment based on candel size
increment = {'day': 10, '4_hour': 10, '1_hour': 40}
# times the candle sticks can be
bartimes = ['day','4_hour','1_hour','15_min'] 

def save_screenshot(image: list,filename: str, ticker: str, chart_type: str, bartime: str):
    filepath = Path('data/images/'+ticker+'/'+chart_type+'/'+bartime+'/'+filename+"_screenshot.png")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as out:
        out.write(image)

def screenshot_chart(ticker: str, bartime: str, chart_type: str):
    if __name__ == '__main__':
        # sets ticker to lower case
        ticker = ticker.lower()
        col_labels = ['date','open','high','low','close','volume']
        # Columns: time | open | high | low | close | volume
        try:
            df = pd.read_csv('data/formatted/'+ticker +'/'+ticker+'_'+bartime+'_data_formatted.csv')
        except Exception:
            return # file does not exist, skip
        chart = Chart()
        chart.crosshair('hidden')
        chart.grid(False,False)
        # set the first set on the chart
        chart.set(df[col_labels].iloc[0:interval[bartime]])
        chart.show()
        image = chart.screenshot()
        filename = (ticker + '_' + bartime + '_' + chart_type + '_' + bartime + 
                    '_' + '1_to_' + bartime + '_' + str(interval[bartime]))
        save_screenshot(image, filename, ticker, chart_type, bartime)       
        # intialize variables to keep track of updates
        count, start_idx, idx = 1, increment[bartime], interval[bartime]
        while idx < df['date'].size:
            chart.update(df[col_labels].iloc[idx])
            # next increment starting point
            if count == increment[bartime] or idx + 1 == df['date'].size:
                filename = (ticker + '_' + bartime + '_' + chart_type + '_' + bartime + '_' + 
                            str(start_idx) + '_to_' + bartime + '_' + str(idx))
                image = chart.screenshot()
                save_screenshot(image, filename, ticker, chart_type, bartime)
                count = 1
                idx += 1
                start_idx += increment[bartime]
                continue
            idx += 1
            count += 1
            time.sleep(0.01)
            
# chart_types = ['candel','line']
ticker = sys.argv[1]
ticker = ticker.lower()
screenshot_chart(ticker, 'day', 'candle')
screenshot_chart(ticker, '4_hour', 'candle')
screenshot_chart(ticker, '1_hour', 'candle')