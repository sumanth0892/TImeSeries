#Analyzing Google stock data
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from sklearn.preprocessing import MinMaxScaler

#Read the data into a pandas dataframe
#Consider the closing prices
totalData = pd.read_csv('GOOG.csv')
closingPrices = totalData.iloc[:,4:5].values
scaler = MinMaxScaler(feature_range = (0,1))
X = scaler.fit_transform(closingPrices)
n = len(X)

#Consider two models where we use 30 and 60 points to train a model and predict
set1 = []; set2 = []
labels1 = []; labels2 = []
for i in range(30,n):
    set1.append(X[i-30:i,0])
    labels1.append(X[i,0])
for i in range(60,n):
    set2.append(X[i-60:i,0])
    labels2.append(X[i,0])
set1,labels1 = np.array(set1),np.array(labels1)
set2,labels2 = np.array(set2),np.array(labels2)
set1 = np.reshape(set1,(set1.shape[0],set1.shape[1],1))
print(set1.shape)
set2 = np.reshape(set2,(set2.shape[0],set2.shape[1],1))
print(set2.shape)

#Model to train the dataset with 30 days prediction
model = Sequential()
model.add(LSTM(50,return_sequences = True,input_shape = (set1.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(50,return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(50,return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss = 'mae',optimizer = 'rmsprop')
h1 = model.fit(set1,labels1,epochs = 100,batch_size = 32)
loss1 = h1.history['loss']
epochs = range(1,len(loss1) + 1)


#Model to train the dataset with 60 days prediction
model1 = Sequential()
model1.add(LSTM(50,return_sequences = True,input_shape = (set2.shape[1],1)))
model1.add(Dropout(0.2))
model1.add(LSTM(50,return_sequences = True))
model1.add(Dropout(0.2))
model1.add(LSTM(50,return_sequences = True))
model1.add(Dropout(0.2))
model1.add(LSTM(50))
model1.add(Dense(1))
model1.compile(loss = 'mae',optimizer = 'rmsprop')
h2 = model1.fit(set2,labels2,epochs = 100,batch_size = 32)
loss2 = h2.history['loss']

#Compare the results
plt.plot(epochs,loss1,'bo',label='Training loss with 30 days lookback')
plt.plot(epochs,loss2,'ro',label='Training loss with 60 days lookback')
plt.xlabel('Epochs')
plt.ylabel('Training losses')
plt.grid(True)
plt.legend(loc = 'best')
plt.show()

#Prepare the testing set
completeTestingSet = pd.read_csv('GOOGTest.csv')
testProcessing = completeTestingSet.iloc[:,4:5].values
total = pd.concat((totalData['Close'],completeTestingSet['Close']),axis = 0)
test_inputs1 = total[len(total) - len(completeTestingSet) - 30:].values
test_inputs2 = total[len(total) - len(completeTestingSet) - 60:].values
test_inputs1 = test_inputs1.reshape(-1,1)
test_inputs2 = test_inputs2.reshape(-1,1)
test_inputs1 = scaler.transform(test_inputs1)
test_inputs2 = scaler.transform(test_inputs2)
test_features1 = []; test_features2 = []
for i in range(30,46):
    test_features1.append(test_inputs1[i-30:i,0])
for i in range(60,76):
    test_features2.append(test_inputs2[i-60:i,0])
test_features1,test_features2 = np.array(test_features1),np.array(test_features2)
test_features1 = np.reshape(test_features1,(test_features1.shape[0],test_features1.shape[1],1))
test_features2 = np.reshape(test_features2,(test_features2.shape[0],test_features2.shape[1],1))
preds1 = model.predict(test_features1)
preds2 = model1.predict(test_features2)
errors = preds1 - preds2

predsAndErrors = pd.DataFrame(preds1,preds2,errors)
predsAndErrors.to_csv('GoogleClosingPredictions.csv')









