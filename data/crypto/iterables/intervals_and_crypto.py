import pickle


def main():

    coinbase_crypto_tickers = [
        "BTCUSD", "ETHUSD", "XRPUSD", "BCHUSD", "LTCUSD", "EOSUSD", "DASHUSD", "LINKUSD", "ZRXUSD", "ETCUSD",
        "XTZUSD", "OXTUSD", "REPUSD", "ATOMUSD", "ALGOUSD", "DNTUSD", "BATUSD", "CVCUSD", "MANAUSD", "KNCUSD",
        "LOOMUSD", "GNTUSD", "NUUSD", "MKRUSD", "NMRUSD", "OMGUSD", "RLCUSD", "BNTUSD", "SNXUSD", "COMPUSD",
        "UMAUSD", "YFIUSD", "RENUSD", "BALUSD", "WBTCUSD", "AAVEUSD", "FILUSD", "ADAUSD", "GRTUSD",
        "SOLUSD", "LRCUSD", "CGLDUSD", "CTSIUSD", "SKLUSD", "SUSHIUSD", "ANKRUSD", "1INCHUSD", "ENJUSD",
        "NKNUSD", "OGNUSD", "CHZUSD", "RLYUSD", "TRBUSD", "BONDUSD", "KEEPUSD", "MLNUSD",
        "TRUUSD", "AMPUSD", "FORTHUSD", "GHSTUSD", "TBTCUSD", "MASKUSD", "AUDIOUSD", "GTCUSD",
        "MIRUSD", "AXSUSD", "LPTUSD", "FLOWUSD", "FARMUSD", "DAIUSD"
    ]

    time_intervals = ['1_day','4_hour','1_hour', '15_minute', '5_minute']


    with open('data/crypto/iterables/coinbase_tickers.pkl', 'wb') as fp:
        pickle.dump(coinbase_crypto_tickers, fp)
        
    with open('data/crypto/iterables/time_intervals.pkl', 'wb') as fp:
        pickle.dump(time_intervals, fp)

if __name__=='__main__':
    main()