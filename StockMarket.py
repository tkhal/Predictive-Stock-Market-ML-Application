# -*- coding: utf-8 -*-
"""SM"""

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

df = web.DataReader('MSFT', data_source='yahoo', start='2014-11-14', end='2020-7-8') # using apple as an example, change start/end dates for data set variables
df

# get the number of rows and cols in the data set
df.shape

#visualize the closing price history of the given stock
plt.figure(figsize=(16,8))
plt.title("Close Price History")
plt.plot(df['Close'])
plt.xlabel("Date", fontsize = 18)
plt.ylabel("Close Price USD ($)", fontsize=18)
plt.show()

#create a new data frame with only the "close column"
data = df.filter(['Close'])
#convert the dataframe to a numpy array
dataset = data.values
#get the number of rows to train the LSTM Model
training_data_len = math.ceil(len(dataset) * .8) # to give 80%
beyond_data_len = math.ceil(len(dataset)* 1.2)
training_data_len
beyond_data_len

# scale the data | Why scale the data? It's always advantageous to apply preprocessing transformations, scaling, or normalizations to the input data before it's presented to a neural network. TLDR, good practice for ML
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset) # range = 0 <= x <= 1
scaled_data

# Create the training data set
train_data = scaled_data[0:training_data_len, : ]
#split the data into x_train and y_train 
x_train = [] # independent training features
y_train = [] # target variable

for i in range(30, len(train_data)):
  x_train.append(train_data[i - 30:i, 0])
  y_train.append(train_data[i, 0])

  if i <= 30: # change to 61 and see what happens (note the differences in the ending numbers for the 1st and 2nd datasets, and how they comapre)
    print(x_train)
    print(y_train)
    print() # new line

# when you run, the x data set (first 60 values) contains the past 60 values, or the history of a stock. The y data set (the last value, the 61st) contains the predicted data, based off the history (x-data set)

# convert the x_train & y_train to numpy arrays, to train them for the LSTM model
x_train, y_train = np.array(x_train), np.array(y_train)

# reshape the data | default output is 2 dimensional, LSTM Model expects 3 dimensions
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

# build the LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# create the testing dataset 
# create a new array containing scaled values from index 1619 - 2003

#this will be the scaled testing dataset

test_data = scaled_data[training_data_len - 30: ]
#create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(30, len(test_data)):
  x_test.append(test_data[i - 30:i, 0])

# convert the data to a numpy array

x_test = np.array(x_test)

# reshape the data
#again, we need to convert from 2d to 3d

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# reshape(xtest, #rows, #columns, # of features)

# get the models predicted price values

predictions = model.predict(x_test) # we want this to be the exact same values in the y_test dataset once we inverse transform the data

predictions = scaler.inverse_transform(predictions) # we want predictions to contain the same values as the y_test dataset. Getting predictions based of x_test dataset

# get the root mean squared error (RMSE) - a good measure of how accurate the model is


rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
rmse # output will give you % error. To get more accurate predictions, increase the epoch

# plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
pred = data[beyond_data_len:]

valid['Predictions'] = predictions
#visualize the model

plt.figure(figsize=(30, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)

plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc= 'lower right')

plt.show()

# show the valid and predicted prices
valid

# Predict the closing stock of a company on a given day

# Get the Quote (example - AAPL)

quote = web.DataReader('MSFT', data_source='yahoo', start= '2014-11-14', end='2019-7-15')

# Create a new data frame
new_df = quote.filter(['Close'])
# Get the last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-30:].values
# Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
# Create an empty list
X_test = []
# Append the last 60 days to the X_test list
X_test.append(last_60_days_scaled)
# convert the X_test data set to numpy array
X_test = np.array(X_test)
# reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

