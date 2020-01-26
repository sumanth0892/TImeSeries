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

def getModel(units,n,dropout_val):
    model = Sequential()
    model.add(LSTM(units,return_sequences = True,input_shape = (n,1)))
    model.add(Dropout(dropout_val))
    model.add(LSTM(units,return_sequences = True))
    model.add(Dropout(dropout_val))
    model.add(LSTM(units,return_sequences = True))
    model.add(Dropout(dropout_val))
    model.add(LSTM(units))
    model.add(Dense(1))
    model.compile(loss = 'mae',optimizer = 'adam')
    return model

def prepTrainingData(data,timeSteps):
    #Assume the data is given as a pandas dataframe
    dataSet = []; labels = []; n = len(data)
    for i in range(timeSteps,n):
        dataSet.append(data[i-timeSteps:i,0])
        labels.append(data[i,0])
    dataSet,labels = np.array(dataSet),np.array(labels)
    dataSet = np.reshape(dataSet,(dataSet.shape[0],dataSet.shape[1],1))
    return dataSet,labels,n

def prepTestingData(tS):
    completeTestingSet = pd.read_csv('GOOGTest.csv')
    testProcessing = completeTestingSet.iloc[:,4:5].values
    total = pd.concat((totalData['Close'],completeTestingSet['Close']),axis = 0)
    test_inputs = total[len(total) - len(completeTestingSet) - tS:].values
    test_inputs = test_inputs.reshape(-1,1)
    test_inputs = scaler.transform(test_inputs)
    test_features = []
    for i in range(tS,tS+16):
        test_features.append(test_inputs[i-tS:i,0])
    test_features = np.array(test_features)
    test_features = np.reshape(test_features,(test_features.shape[0],test_features.shape[1],1))
    return test_features
 
    

stepsArray = [15,30,45,60,75,90]
maxError = []; predictions = []; losses = []
for timeStep in stepsArray:
    data,labels,n = prepTrainingData(X,timeStep)
    model = getModel(50,timeStep,0.2)
    history = model.fit(data,labels,epochs = 10,batch_size = 128)
    losses.append(history.history['loss'])
    maxError.append(max(history.history['loss']))
    test_features = prepTestingData(timeStep)
    preds = model.predict(test_features)
    predictions.append(scaler.inverse_transform(preds))
    
    
#Prepare the testing set
predsAndErrors = pd.DataFrame(np.column_stack((pred for pred in predictions)),columns=['15 days','30 days','45 days','60 days','75 days','90 days'])
predsAndErrors.to_csv('Predictions.csv')










