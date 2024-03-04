from hmm_training import StockPredictor
import pickle

data = pickle.load(open('mapes.pkl', 'rb'))
data1 = pickle.load(open('mapes1.pkl', 'rb'))
data2 = pickle.load(open('mapes2.pkl', 'rb'))
data3 = pickle.load(open('mapes3.pkl', 'rb'))
data4 = pickle.load(open('mapes4.pkl', 'rb'))

print(data3)


# mapes = [data1,data2,data3,data4]

# print(len(data))

# for arr in mapes:
#     for elem in arr:
#         if elem in data:
#             continue
#         data.append(elem)

# print(len(data))

# print(data)


# stockpred = StockPredictor(('zts', '1_day'), multimodal=False)
# stockpred.fit()
# mape = stockpred.predict_close_prices_for_days(252, with_plot=True)
# print(mape)

