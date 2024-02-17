import pandas as pd
from lightweight_charts import Chart
from pathlib import Path
import sys

def screenshot_chart(ticker: str, filename: str, chart_type: str, df: pd.DataFrame):
    if __name__ == '__main__':
        chart = Chart()
        chart.set(df)
        chart.show()
        image = chart.screenshot()
        filepath = Path('data/images/'+ticker+'/'+
                        chart_type+'/'+filename+"_screenshot.png")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as out:
            out.write(image)

def screenshots_for_ticker(ticker: str, bartime: str, chart_type: str):
    col_labels = ['date','open','high','low','close','volume']
    # Columns: time | open | high | low | close | volume
    try:
        df = pd.read_csv('data/formatted/'+ticker +'/'+ticker+'_'+bartime+'_data_formatted.csv')
    except Exception:
        return # file does not exist, skip
    count = 1
    start_idx, next_idx, idx = 0, 0, 0
    while idx < df['date'].size:
        # next increment starting point
        if count == increment[bartime]:
            next_idx = idx
        # end of interval, or end of frame
        if count == interval[bartime] or idx + 1 == df['date'].size:
            filename = (ticker + '_' + bartime + '_' + chart_type + '_day' +
                        str(start_idx + 1) + '_to_day' + str(idx + 1))
            screenshot_chart(ticker, filename, chart_type, df[col_labels].iloc[start_idx:idx])
            count = 0
            start_idx = next_idx
            # reached end of dataframe
            if idx + 1 == df['date'].size:
                break
            idx = start_idx
        idx += 1
        count += 1


# interval of viewing the chart
interval = {'day': 120, '4_hour': 30, '1_hour': 10}
# amount to increment based on candel size
increment = {'day': 10, '4_hour': 40, '1_hour': 24}

bartimes = ['day','4_hour','1_hour','15_min'] 
# chart_types = ['candel','line']
screenshots_for_ticker(sys.argv[1], 'day', 'candle')