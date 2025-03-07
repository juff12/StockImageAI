import pickle


def main():
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

    nasdaq_100_tickers = [
        "AAPL", "ADBE", "ADI", "ADP", "ADSK", "ALGN", "ALXN", "AMAT", "AMD", "AMGN",
        "ANSS", "ASML", "ATVI", "AVGO", "BIDU", "BIIB", "BKNG", "BMRN", "CDNS", "CDW",
        "CERN", "CHKP", "CHTR", "CMCSA", "COST", "CPRT", "CSCO", "CSX", "CTAS", "CTSH",
        "CTXS", "DLTR", "DOCU", "DXCM", "EA", "EBAY", "EXC", "EXPE", "FAST", "FB",
        "FISV", "FOX", "FOXA", "GILD", "GOOG", "GOOGL", "IDXX", "ILMN", "INCY", "INTC",
        "INTU", "ISRG", "JD", "KDP", "KHC", "KLAC", "LRCX", "LULU", "LUMN", "LVGO",
        "MAR", "MCHP", "MDLZ", "MELI", "MNST", "MRNA", "MSFT", "MU", "MXIM", "NFLX",
        "NTAP", "NTES", "NVDA", "NXPI", "OKTA", "ORLY", "PAYX", "PCAR", "PDD", "PEP",
        "PYPL", "QCOM", "REGN", "ROST", "SBUX", "SGEN", "SIRI", "SNPS", "SPLK", "SWKS",
        "TCOM", "TEAM", "TMUS", "TSLA", "TXN", "VRSK", "VRTX", "WBA", "WDAY", "XEL",
        "XLNX", "ZM"
    ]

    time_intervals = ['1_day','4_hour','1_hour']

    with open('iterables/sp500_tickers.pkl', 'wb') as fp:
        pickle.dump(sp500_tickers, fp)
        
    with open('iterables/nasdaq_100_tickers.pkl', 'wb') as fp:
        pickle.dump(nasdaq_100_tickers, fp)
        
    with open('iterables/time_intervals.pkl', 'wb') as fp:
        pickle.dump(time_intervals, fp)

if __name__=='__main__':
    main()