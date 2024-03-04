from hmm_training import StockPredictor
import pickle
from pathlib import Path
import pandas as pd

data = pickle.load(open('mapes.pkl', 'rb'))
# data1 = pickle.load(open('mapes1.pkl', 'rb'))
# data2 = pickle.load(open('mapes2.pkl', 'rb'))
# data3 = pickle.load(open('mapes3.pkl', 'rb'))
# data4 = pickle.load(open('mapes4.pkl', 'rb'))

#print(data4)


#mapes = [data1,data2,data3,data4]

print(len(data))

# for arr in mapes:
#     for elem in arr:
#         if elem in data:
#             continue
#         data.append(elem)

# with open('mapes.pkl', 'wb') as file:
#     pickle.dump(data, file)

df_mape = pd.DataFrame(data, columns=['ticker','1_day','4_hour','1_hour'])
filepath = Path('data/mapes/sp500_mapes.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
df_mape.to_csv(filepath,index=False)


# stockpred = StockPredictor(('zts', '1_day'), multimodal=False)
# stockpred.fit()
# mape = stockpred.predict_close_prices_for_days(252, with_plot=True)
# print(mape)

