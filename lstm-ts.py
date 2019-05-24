import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model

# %matplotlib inline
# Loading DataFrame file
stock_data = pd.read_csv("./AA.csv")
# Creating the new feature is average of "high" and "low" value
stock_data["average"] = (stock_data["high"] + stock_data["low"])/2
# Select 2 feature on DataFrame to Dataset
input_feature = stock_data.iloc[:,[1,6]].values
input_data = input_feature

# Visualization the Dataset
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

# Using MinMaxScaler module (Features Scaling module on Scikit Learn) to scale Dataset values
sc = MinMaxScaler(feature_range=(0,1))
input_data[:,0:2] = sc.fit_transform(input_feature[:,:])

# Setting the memory value of LSTM Cell 
lookback = 50

# Select 30% amount currently of Data to Testset, 70% older Data to Training set.
test_size = int(.3 * len(stock_data))
X = []
y = []
# Reconstructing the Dataset to real Dataset to implement
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

# Select training model is RNN with Sequential module on Keras
model = Sequential()
# Initializa the parameters of Training
n_units = 30        # number of neurals on a layer
n_epochs = 1      # number of times to running Training
n_batch_size = 32   # size of batch per time of Gradient Descent
# NOTE:
# 1. Number of calculation must be large enough to GPU computing will be faster CPU computing
# 2. Size of Batch should be exponent of 2 if computing on GPU. Because architecture of GPU is grid, compatible with 2^N
# 3. Number of hidden layers should be 2-4; to avoid overfitting

# Generate the model of RNN
model.add(LSTM(units = n_units, return_sequences = True, input_shape = (X.shape[1],2)))     # Input layer
model.add(LSTM(units = n_units, return_sequences = True))       # Hidden 1 layer
model.add(LSTM(units = n_units))        # Hidden 2 layer
model.add(Dense(units = 1))     # Output layer
model.summary()
# Select optimize parameter: "Adam Optimizer" and "Mean Square Loss"
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Training model with Training set {Features, Label}
model.fit(X[3731:,:,:], y[3731:,], epochs = n_epochs, batch_size = n_batch_size)

# Save the model
model.save("./AA_model.h5")

# Implement result of model to Test set
predicted_value = model.predict(X_test)
# Show the plot of result
plt.plot(predicted_value, color = 'red')
plt.plot(input_data[lookback:test_size+(2*lookback),1], color = 'green')
plt.title("Opening price of stocks sold with n_units = " + str(n_units) + ", n_epochs = " + str(n_epochs) + ", n_batch_size = " + str(n_batch_size))
plt.xlabel("Time (latest-> oldest)")
plt.ylabel("Stock Opening Price")
plt.show()

