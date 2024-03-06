import pickle


def main():

    coinbase_crypto_tickers = [
        "BTCUSD", "ETHUSD", "XRPUSD", "BCHUSD", "LTCUSD", "EOSUSD", "DASHUSD", "LINKUSD", "ZRXUSD", "ETCUSD",
        "XTZUSD", "OXTUSD", "REPUSD", "ATOMUSD", "ALGOUSD", "DNTUSD", "BATUSD", "CVCUSD", "MANAUSD", "KNCUSD",
        "LOOMUSD", "GNTUSD", "NUUSD", "MKRUSD", "NMRUSD", "OMGUSD", "RLCUSD", "BNTUSD", "SNXUSD", "COMPUSD",
        "UMAUSD", "YFIUSD", "RENUSD", "BALUSD", "WBTCUSD", "NUUSD", "AAVEUSD", "FILUSD", "ADAUSD", "GRTUSD",
        "SOLUSD", "LRCUSD", "CGLDUSD", "CTSIUSD", "SKLUSD", "SUSHIUSD", "ANKRUSD", "1INCHUSD", "ENJUSD",
        "NKNUSD", "OGNUSD", "CHZUSD", "RLYUSD", "NUUSD", "TRBUSD", "BONDUSD", "KEEPUSD", "MLNUSD", "BNTUSD",
        "TRUUSD", "AMPUSD", "RVNUSD", "FORTHUSD", "GHSTUSD", "TBTCUSD", "MASKUSD", "AUDIOUSD", "GTCUSD",
        "MIRUSD", "AXSUSD", "LPTUSD", "FLOWUSD", "ERNUSD", "FARMUSD", "HNSUSD", "DAIUSD", "SKLUSD", "1INCHUSD",
        "RLYUSD", "NUUSD", "BONDUSD", "BNTUSD", "MLNUSD", "KEEPUSD", "NUUSD", "TRUUSD", "AMPUSD", "TRBUSD",
        "GTCUSD", "RVNUSD", "ERNUSD", "FARMUSD", "HNSUSD", "FLOWUSD", "DAIUSD", "GHSTUSD", "AUDIOUSD", "FORTHUSD",
        "TBTCUSD", "MASKUSD", "AXSUSD", "MIRUSD", "LPTUSD", "ANKRUSD", "SUSHIUSD", "NKNUSD", "ENJUSD", "1INCHUSD",
        "SKLUSD", "CTSIUSD", "CGLDUSD", "LRCUSD", "SOLUSD", "GRTUSD", "ADAUSD", "FILUSD", "AAVEUSD", "NUUSD",
        "WBTCUSD", "BALUSD", "RENUSD", "YFIUSD", "UMAUSD", "COMPUSD", "SNXUSD", "BNTUSD", "RLCUSD", "OMGUSD",
        "NMRUSD", "MKRUSD", "NUUSD", "GNTUSD", "LOOMUSD", "KNCUSD", "MANAUSD", "CVCUSD", "BATUSD", "DNTUSD",
        "ALGOUSD", "ATOMUSD", "REPUSD", "OXTUSD", "XTZUSD", "ETCUSD", "ZRXUSD", "LINKUSD", "DASHUSD", "EOSUSD",
        "LTCUSD"
    ]

    time_intervals = ['1_day','4_hour','1_hour', '15_min', '5_min']


    with open('crypto/iterables/coinbase_tickers.pkl', 'wb') as fp:
        pickle.dump(coinbase_crypto_tickers, fp)
        
    with open('crypto/iterables/time_intervals.pkl', 'wb') as fp:
        pickle.dump(time_intervals, fp)

if __name__=='__main__':
    main()