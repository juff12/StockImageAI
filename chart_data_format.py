import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import logging

sp500_tickers = [
        "mmm", "abt", "abbv", "acn", "atvi", "adbe", "amd", "aes", "afl", "a",
        "akam", "alk", "alb", "are", "algn", "alle", "lnt", "all", "googl",
        "goog", "mo", "amzn", "amcr", "aee", "aal", "aep", "axp", "aig",
        "amt", "awk", "amp", "abc", "ame", "amgn", "aph", "adi", "antm",
        "aon", "aos", "apa", "aapl", "amat", "aptv", "adm", "anet", "ajg",
        "aiz", "t", "ato", "adsk", "adp", "azo", "avb", "avy", "bkr",
        "bll", "bac", "bk", "bax", "bdx", "brk.b", "bby", "bio", "biib",
        "blk", "ba", "bkng", "bwa", "bxp", "bsx", "bmy", "avgo", "br",
        "bf.b", "chrw", "cog", "cdns", "czr", "cpb", "cof", "cah", "kmx",
        "ccl", "carr", "ctlt", "cat", "cboe", "cbre", "cdw", "ce", "cnc",
        "cnp", "cern", "cf", "crl", "schw", "chtr", "cvx", "cmg", "cb",
        "chd", "ci", "cinf", "ctas", "csco", "c", "cfg", "ctxs", "clx",
        "cme", "cms", "ko", "ctsh", "cl", "cmcsa", "cma", "cag", "cop",
        "ed", "stz", "coo", "cprt", "glw", "ctva", "cost", "cci", "csx",
        "cmi", "cvs", "dhi", "dhr", "dri", "dva", "de", "dal", "xray",
        "dvn", "dxcm", "fang", "dlr", "dfs", "disca", "disck", "dish", "dg",
        "dltr", "d", "dpz", "dov", "dow", "dte", "duk", "dre", "dd",
        "dxc", "emn", "etn", "ebay", "ecl", "eix", "ew", "ea", "emr",
        "enph", "etr", "eog", "efx", "eqix", "eqr", "ess", "el", "etsy",
        "evrg", "es", "re", "exc", "expe", "expd", "exr", "xom", "ffiv",
        "fb", "fast", "frt", "fdx", "fis", "fitb", "fe", "frc", "fisv",
        "flt", "flir", "fls", "fmc", "f", "ftnt", "ftv", "fbhs", "foxa",
        "fox", "ben", "fcx", "gps", "grmn", "it", "gnrc", "gd", "ge",
        "gis", "gm", "gpc", "gild", "gl", "gpn", "gs", "gww", "hal",
        "hbi", "hig", "has", "hca", "peak", "hsic", "hsy", "hes", "hpe",
        "hlt", "hfc", "holx", "hd", "hon", "hrl", "hst", "hwm", "hpq",
        "hum", "hban", "hii", "iex", "idxx", "info", "itw", "ilmn", "incy",
        "ir", "intc", "ice", "ibm", "ip", "ipg", "iff", "intu", "isrg",
        "ivz", "ipgp", "iqv", "irm", "jkhy", "j", "jbht", "sjm", "jnj",
        "jci", "jpm", "jnpr", "ksu", "k", "key", "keys", "kmb", "kim",
        "kmi", "klac", "khc", "kr", "lb", "lhx", "lh", "lrcx", "lw",
        "lvs", "leg", "ldos", "len", "lly", "lnc", "lin", "lyv", "lkq",
        "lmt", "l", "low", "lumn", "lyb", "mtb", "mro", "mpc", "mktx",
        "mar", "mmc", "mlm", "mas", "ma", "mkc", "mxim", "mcd", "mck",
        "mdt", "mrk", "met", "mtd", "mgm", "mchp", "mu", "msft", "maa",
        "mhk", "tap", "mdlz", "mpwr", "mnst", "mco", "ms", "mos", "msi",
        "msci", "ndaq", "ntap", "nflx", "nwl", "nem", "nwsa", "nws", "nee",
        "nlsn", "nke", "ni", "nsc", "ntrs", "noc", "nlok", "nclh", "nov",
        "nrg", "nue", "nvda", "nvr", "nxpi", "orly", "oxy", "odfl", "omc",
        "oke", "orcl", "otis", "pcar", "pkg", "ph", "payx", "payc", "pypl",
        "penn", "pnr", "pbct", "pep", "pki", "prgo", "pfe", "pm", "psx",
        "pnw", "pxd", "pnc", "pool", "ppg", "ppl", "pfg", "pg", "pgr",
        "pld", "pru", "ptc", "peg", "psa", "phm", "pvh", "qrvo", "pwr",
        "qcom", "dgx", "rl", "rjf", "rtx", "o", "reg", "regn", "rf",
        "rsg", "rmd", "rhi", "rok", "rol", "rop", "rost", "rcl", "spgi",
        "crm", "sbac", "slb", "stx", "see", "sre", "now", "shw", "spg",
        "swks", "sna", "so", "luv", "swk", "sbux", "stt", "ste", "syk",
        "sivb", "syf", "snps", "syy", "tmus", "trow", "ttwo", "tpr", "tgt",
        "tel", "tdy", "tfx", "ter", "tsla", "txn", "txt", "tmo", "tjx",
        "tsco", "tt", "tdg", "trv", "tfc", "twtr", "tyl", "tsn", "udr",
        "ulta", "usb", "uaa", "ua", "unp", "ual", "unh", "ups", "uri",
        "uhs", "unm", "vlo", "vtr", "vrsn", "vrsk", "vz", "vrtx", "vfc",
        "viac", "vtrs", "v", "vnt", "vno", "vmc", "wrb", "wab", "wmt",
        "wba", "dis", "wm", "wat", "wec", "wfc", "well", "wst", "wdc",
        "wu", "wrk", "wy", "whr", "wmb", "wltw", "wynn", "xel", "xlnx",
        "xyl", "yum", "zbra", "zbh", "zion", "zts"
    ]
time_intervals = ['1_day','4_hour','1_hour']

def _init_logger():
    _logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(logging.DEBUG)
    return _logger

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
    
    # out file path
    filepath = Path('data/formatted/'+ticker+'/'+ticker+"_"+time_interval+"_data_formatted.csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)

def main():
    logger = _init_logger()
    logger.info('>>> Begining Stock Data Formatting')
    for ticker in tqdm(sp500_tickers):
        for time_interval in time_intervals:
            format_data(ticker.lower(), time_interval)
    logger.info('Stock Data Formatting Complete <<<')

if __name__ == '__main__':
    main()