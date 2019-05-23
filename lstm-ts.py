import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM

# %matplotlib inline

stock_data = pd.read_csv("./HistoricalQuotes.csv")

stock_data["average"] = (stock_data["high"] + stock_data["low"])/2

input_feature = stock_data.iloc[:,[2,6]].values
input_data = input_feature

# plt.plot(input_feature[:,0])
# plt.title("Volume of stocks sold")
# plt.xlabel("Time (latest-> oldest)")
# plt.ylabel("Volume of stocks traded")
# plt.show()

# plt.plot(input_feature[:,1], color='blue')
# plt.title("Google Stock Prices")
# plt.xlabel("Time (latest-> oldest)")
# plt.ylabel("Stock Opening Price")
# plt.show()

sc = MinMaxScaler(feature_range=(0,1))
input_data[:,0:2] = sc.fit_transform(input_feature[:,:])

lookback = 50

test_size = int(.3 * len(stock_data))
X = []
y = []
for i in range(len(stock_data) - lookback - 1):
    t = []
    for j in range(0,lookback):
        t.append(input_data[[(i+j)], :])
    X.append(t)
    y.append(input_data[i + lookback,1])

X, y = np.array(X), np.array(y)
X_test = X[:test_size + lookback]
X = X.reshape(X.shape[0],lookback, 2)
X_test = X_test.reshape(X_test.shape[0],lookback, 2)
print(X.shape)
print(X_test.shape)


model = Sequential()
n_units = 10
n_epochs = 200
n_batch_size = 32
model.add(LSTM(units = n_units, return_sequences = True, input_shape = (X.shape[1],2)))
model.add(LSTM(units = n_units, return_sequences = True))
model.add(LSTM(units = n_units))
model.add(Dense(units = 1))
model.summary()

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X[428:,:,:], y[428:,], epochs = n_epochs, batch_size = n_batch_size)

# weights, biases = model.layers[0].get_weights()
# print(weights)
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print(biases)

predicted_value = model.predict(X_test)

plt.plot(predicted_value, color = 'red')
plt.plot(input_data[lookback:test_size+(2*lookback),1], color = 'green')
plt.title("Opening price of stocks sold with n_units = " + str(n_units) + ", n_epochs = " + str(n_epochs) + ", n_batch_size = " + str(n_batch_size))
plt.xlabel("Time (latest-> oldest)")
plt.ylabel("Stock Opening Price")
plt.show()

