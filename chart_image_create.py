import pandas as pd
from lightweight_charts import Chart
from pathlib import Path
import sys
import time

# interval of viewing the chart

# test values (interval/increment/barspacing) day: 140/10/3, 4_hour: 400/15/1.75, 1_hour: 500/30/1.5
interval = {'1_day': 140, '4_hour': 120, '1_hour': 120}
# amount to increment based on candel size
increment = {'1_day': 10, '4_hour': 15, '1_hour': 20}
# times the candle sticks can be
bartimes = ['1_day','4_hour','1_hour','15_min'] 
# bar spacing prests -> fits max zoom -> lower value = more zoom out, higher value = less zoom out
barspacing = {'1_day': 3, '4_hour': 1.75, '1_hour': 1.5}


def save_screenshot(image: list,filename: str, ticker: str, chart_type: str, bartime: str):
    filepath = Path('data/images/'+ticker+'/'+chart_type+'/'+bartime+'/'+filename+"_screenshot.png")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as out:
        out.write(image)

def screenshot_chart(ticker: str, bartime: str, chart_type: str):    
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
    chart.time_scale(min_bar_spacing=barspacing[bartime]) ####################
    
    chart.set(df[col_labels].iloc[0:interval[bartime]]) ####################
    chart.fit()
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
        time.sleep(0.005)
    chart.exit()

# main code of script
def main():
    # chart_types = ['candel','line']
    ticker = sys.argv[1]
    ticker = ticker.lower()
    screenshot_chart(ticker, '1_day', 'candle')
    screenshot_chart(ticker, '4_hour', 'candle')
    screenshot_chart(ticker, '1_hour', 'candle')

# run script
if __name__ == '__main__':
    main()