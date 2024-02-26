import pandas as pd
from datetime import datetime
from pathlib import Path


sp500_tickers = [
    "MMM", "ABT", "ABBV", "ACN", "ATVI", "ADBE", "AMD", "AES", "AFL", "A",
    "APD", "AKAM", "ALK", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
    "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP", "AXP", "AIG",
    "AMT", "AWK", "AMP", "ABC", "AME", "AMGN", "APH", "ADI", "ANSYS", "ANTM",
    "AON", "AOS", "APA", "AAPL", "AMAT", "APTV", "ADM", "ANET", "AJG",
    "AIZ", "T", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "BKR",
    "BLL", "BAC", "BK", "BAX", "BDX", "BRK.B", "BBY", "BIO", "BIIB",
    "BLK", "BA", "BKNG", "BWA", "BXP", "BSX", "BMY", "AVGO", "BR",
    "BF.B", "CHRW", "COG", "CDNS", "CZR", "CPB", "COF", "CAH", "KMX",
    "CCL", "CARR", "CTLT", "CAT", "CBOE", "CBRE", "CDW", "CE", "CNC",
    "CNP", "CERN", "CF", "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB",
    "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CTXS", "CLX",
    "CME", "CMS", "KO", "CTSH", "CL", "CMCSA", "CMA", "CAG", "COP",
    "ED", "STZ", "COO", "CPRT", "GLW", "CTVA", "COST", "CCI", "CSX",
    "CMI", "CVS", "DHI", "DHR", "DRI", "DVA", "DE", "DAL", "XRAY",
    "DVN", "DXCM", "FANG", "DLR", "DFS", "DISCA", "DISCK", "DISH", "DG",
    "DLTR", "D", "DPZ", "DOV", "DOW", "DTE", "DUK", "DRE", "DD",
    "DXC", "EMN", "ETN", "EBAY", "ECL", "EIX", "EW", "EA", "EMR",
    "ENPH", "ETR", "EOG", "EFX", "EQIX", "EQR", "ESS", "EL", "ETSY",
    "EVRG", "ES", "RE", "EXC", "EXPE", "EXPD", "EXR", "XOM", "FFIV",
    "FB", "FAST", "FRT", "FDX", "FIS", "FITB", "FE", "FRC", "FISV",
    "FLT", "FLIR", "FLS", "FMC", "F", "FTNT", "FTV", "FBHS", "FOXA",
    "FOX", "BEN", "FCX", "GPS", "GRMN", "IT", "GNRC", "GD", "GE",
    "GIS", "GM", "GPC", "GILD", "GL", "GPN", "GS", "GWW", "HAL",
    "HBI", "HIG", "HAS", "HCA", "PEAK", "HSIC", "HSY", "HES", "HPE",
    "HLT", "HFC", "HOLX", "HD", "HON", "HRL", "HST", "HWM", "HPQ",
    "HUM", "HBAN", "HII", "IEX", "IDXX", "INFO", "ITW", "ILMN", "INCY",
    "IR", "INTC", "ICE", "IBM", "IP", "IPG", "IFF", "INTU", "ISRG",
    "IVZ", "IPGP", "IQV", "IRM", "JKHY", "J", "JBHT", "SJM", "JNJ",
    "JCI", "JPM", "JNPR", "KSU", "K", "KEY", "KEYS", "KMB", "KIM",
    "KMI", "KLAC", "KHC", "KR", "LB", "LHX", "LH", "LRCX", "LW",
    "LVS", "LEG", "LDOS", "LEN", "LLY", "LNC", "LIN", "LYV", "LKQ",
    "LMT", "L", "LOW", "LUMN", "LYB", "MTB", "MRO", "MPC", "MKTX",
    "MAR", "MMC", "MLM", "MAS", "MA", "MKC", "MXIM", "MCD", "MCK",
    "MDT", "MRK", "MET", "MTD", "MGM", "MCHP", "MU", "MSFT", "MAA",
    "MHK", "TAP", "MDLZ", "MPWR", "MNST", "MCO", "MS", "MOS", "MSI",
    "MSCI", "NDAQ", "NTAP", "NFLX", "NWL", "NEM", "NWSA", "NWS", "NEE",
    "NLSN", "NKE", "NI", "NSC", "NTRS", "NOC", "NLOK", "NCLH", "NOV",
    "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC",
    "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PH", "PAYX", "PAYC", "PYPL",
    "PENN", "PNR", "PBCT", "PEP", "PKI", "PRGO", "PFE", "PM", "PSX",
    "PNW", "PXD", "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR",
    "PLD", "PRU", "PTC", "PEG", "PSA", "PHM", "PVH", "QRVO", "PWR",
    "QCOM", "DGX", "RL", "RJF", "RTX", "O", "REG", "REGN", "RF",
    "RSG", "RMD", "RHI", "ROK", "ROL", "ROP", "ROST", "RCL", "SPGI",
    "CRM", "SBAC", "SLB", "STX", "SEE", "SRE", "NOW", "SHW", "SPG",
    "SWKS", "SNA", "SO", "LUV", "SWK", "SBUX", "STT", "STE", "SYK",
    "SIVB", "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO", "TPR", "TGT",
    "TEL", "TDY", "TFX", "TER", "TSLA", "TXN", "TXT", "TMO", "TJX",
    "TSCO", "TT", "TDG", "TRV", "TFC", "TWTR", "TYL", "TSN", "UDR",
    "ULTA", "USB", "UAA", "UA", "UNP", "UAL", "UNH", "UPS", "URI",
    "UHS", "UNM", "VLO", "VTR", "VRSN", "VRSK", "VZ", "VRTX", "VFC",
    "VIAC", "VTRS", "V", "VNT", "VNO", "VMC", "WRB", "WAB", "WMT",
    "WBA", "DIS", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC",
    "WU", "WRK", "WY", "WHR", "WMB", "WLTW", "WYNN", "XEL", "XLNX",
    "XYL", "YUM", "ZBRA", "ZBH", "ZION", "ZTS"
]
time_intervals = ['1_day','4_hour','1_hour']

def fractional_change(open, high, low, close):
    fracChange = (close - open) / open
    fracHigh = (high - open) / open
    fracLow = (open - low) / open
    return [fracChange, fracHigh, fracLow]
    
def trunc_datetime_month(someDate):
    # arbitrary values for setting all values equal except the month
    return someDate.replace(year=2018, day=1, hour=0, minute=0, second=0, microsecond=0)

def format_data(ticker: str, time_interval: str):
    # returns because there is an issue with the file
    try:
        # load raw data frames
        filepath = Path('data/raw/'+ticker+'/'+ticker+"_"+time_interval+"_data_raw.csv")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(filepath)
        # function to convert Unix msec timestamp to datetime (YYYY-MM-DD)
        convert_date = lambda x: datetime.fromtimestamp(x / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
        df['date'] = df['timestamp'].apply(convert_date)
    except Exception:
        # skip, file formated wrong
        return
    # remove uncessary columns
    df = df[['date','open','high','low','close','volume']]
    frac, fracHigh, fracLow = [], [], []
    for i, row in df.iterrows():
        temp = fractional_change(row['open'],row['high'],row['low'],row['close'])
        frac.append(temp[0])
        fracHigh.append(temp[1])
        fracLow.append(temp[2])
    df['frac'] = frac
    df['fracHigh'] = fracHigh
    df['fracLow'] = fracLow
    
    # dictionary of months for fast lookups
    datetime_val = [datetime(2018,i,1) for i in range(1,13)]
    months = dict.fromkeys(month for month in datetime_val)
    months.update((k, i + 1) for i, k in enumerate(months))
    month = []
    # record of current month
    curr_month = trunc_datetime_month(datetime.strptime(df['date'][0], "%Y-%m-%d %H:%M:%S"))
    # label of the current month    
    for date in df['date']:
        if time_interval == '1_hour' or time_interval == '4_hour':
            date = trunc_datetime_month(datetime.strptime(date,'%Y-%m-%d %H:%M:%S'))
        else:
            date = trunc_datetime_month(datetime.strptime(date,'%Y-%m-%d %H:%M:%S'))
        if date == curr_month:
            curr_month = date
        month.append(months[date])
    df['month'] = month
    # out file path
    filepath = Path('data/formatted/'+ticker+'/'+ticker+"_"+time_interval+"_data_formatted.csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)

def main():
    for ticker in sp500_tickers:
        for time_interval in time_intervals:
            format_data(ticker.lower(), time_interval)

if __name__ == '__main__':
    main()