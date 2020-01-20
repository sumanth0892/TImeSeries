import os
import numpy as np
import matplotlib.pyplot as plt
fname = '/Users/sumanth/Documents/PythonLearning/jena_climate_2009_2016.csv'
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(header)
print(len(lines))

D = np.zeros((len(lines),len(header)-1))
for i,line in enumerate(lines):
    value = [float(x) for x in line.split(',')[1:]]
    D[i,:] = value
T = D[:,1]
plt.plot(range(len(T)),T)
plt.show()

#To train, let us use the first 100,000 timesteps as the training data and the
#remaining as testing data
M = D[:100000].mean(axis = 0)
D -= M
S = D[:100000].std(axis=0)
D /= S

def generator(data,lookback,delay,minIndex,maxIndex,
              shuffle = False,batch_size = 128,step = 6):
    if maxIndex is None:
        maxIndex = len(data) - delay - 1
    i = minIndex + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(minIndex + lookback,
                                     maxIndex,
                                     size = batch_size)
        else:
            if i+batch_size >= maxIndex:
                i = minIndex + lookback
            rows = np.arange(i,min(i + batch_size,maxIndex))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback//step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j,row in enumerate(rows):
            indices = range(rows[j] - lookback,rows[j],step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples,targets

lookback = 1440
step = 6
delay = 144
batch_size = 128

trainGen = generator(D,
                     lookback = lookback,
                     delay = delay,
                     minIndex = 0,
                     maxIndex = 100000,
                     shuffle = True,
                     step = step,
                     batch_size = batch_size)

valGen = generator(D,
                   lookback = lookback,
                   delay = delay,
                   minIndex = 100001,
                   maxIndex = 250000,
                   step = step,
                   batch_size = batch_size)

valGen = generator(D,
                   lookback = lookback,
                   delay = delay,
                   minIndex = 250001,
                   maxIndex = None,
                   step = step,
                   batch_size = batch_size)

valSteps = (250000 - 100001 - lookback)//batch_size
testSteps = (len(D) - 250001 - lookback)//batch_size

#A basic ML approach
from keras import models,layers
modelBasic = models.Sequential()
modelBasic.add(layers.Flatten(input_shape=(lookback//step,D.shape[-1])))
modelBasic.add(layers.Dense(32,activation='relu'))
modelBasic.add(layers.Dense(1))
modelBasic.compile(optimizer = 'rmsprop',loss = 'mae')
history = modelBasic.fit_generator(trainGen,
                                   steps_per_epoch = 500,
                                   epochs = 20,
                                   validation_data = valGen,
                                   validation_steps = valSteps)

lossBasic = history.history['loss']
val_lossBasic = history.history['val_loss']
epochs = range(len(lossBasic))
import matplotlib.pyplot as plt
plt.figure()
plt.plot(epochs,lossBasic,'bo',label = 'Training Loss')
plt.plot(epochs,val_lossBasic,'b',label = 'Validation Loss')
plt.title("Training and Validation losses")
plt.legend()
plt.show()

#A first recurrent baseline
modelRec = models.Sequential()
modelRec.add(layers.GRU(32,input_shape = (None,D.shape[-1])))
modelRec.add(layers.Dense(1))
modelRec.compile(optimizer = 'rmsprop',loss = 'mae')
history = modelRec.fit_generator(trainGen,
                                 steps_per_epoch = 500,
                                 epochs = 20,
                                 validation_data = valGen,
                                 validation_steps = valSteps)


lossRec = history.history['loss']
val_lossRec = history.history['val_loss']
epochs = range(len(lossRec))
plt.figure()
plt.plot(epochs,lossRec,'bo',label = 'Training Loss')
plt.plot(epochs,val_lossRec,'b',label = 'Validation Loss')
plt.title("Training and Validation losses")
plt.legend()
plt.show()

#Stacking Recurrent layers
modelStack = models.Sequential()
modelStack.add(layers.GRU(32,dropout = 0.1,
                          recurrent_dropout = 0.5,
                          return_sequences = True,
                          input_shape = (None,D.shape[-1])))
modelStack.add(layers.GRU(64,activation='relu',
                          dropout = 0.1,
                          recurrent_dropout = 0.5))
modelStack.add(layers.Dense(1))
modelStack.compile(optimizer = 'rmsprop',loss = 'mae')
history = modelStack.fit_generator(trainGen,
                                   steps_per_epoch = 500,
                                   epochs = 40,
                                   validation_data = valGen,
                                   validation_steps = valSteps)
lossStack = history.history['loss']
val_lossStack = history.history['val_loss']
epochs = range(len(lossStack))
import matplotlib.pyplot as plt
plt.figure()
plt.plot(epochs,lossStack,'bo',label = 'Training Loss')
plt.plot(epochs,val_lossStack,'b',label = 'Validation Loss')
plt.title("Training and Validation losses")
plt.legend()
plt.show()

#Bidirectional RNNs
def reverseGenerator(data,lookback,minIndex,maxIndex,
                     shuffle = False,batch_size = 128,step = 6):
    if maxIndex is None:
        maxIndex = len(data) - delay - 1
    i = minIndex + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(minIndex + lookback,
                                     maxIndex,size = batch_size)
        else:
            if i + batch_size >= maxIndex:
                i = minIndex + lookback
            rows = np.arange(i,min(i + batch_size,maxIndex))
            i += len(rows)
        samples = np.zeros((len(rows),
                            lookback//step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j,row in enumerate(rows):
            indices = range(rows[j] - lookback,rows[j],step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples[:,::-1,:],targets

trainGenRev = reverseGenerator(D,
                               lookback = lookback,
                               delay = delay,
                               minIndex = 0,
                               maxIndex = 100000,
                               shuffle = True,
                               step = step,
                               batch_size = batch_size)
valGenRev = reverseGenerator(D,
                             lookback = lookback,
                             delay = delay,
                             minIndex = 100001,
                             maxIndex = 250000,
                             step = step,
                             batch_size = batch_size)
modelBidirec = models.Sequential()
modelBidirec.add(layers.GRU(32,input_shape = (None,D.shape[-1])))
modelBidirec.add(layers.Dense(1))
modelBidirec.compile(optimizer = 'rmsprop',loss = 'mae')
history = modelBidirec.fit_generator(trainGenRev,
                                     steps_per_epoch = 500,
                                     epochs = 20,
                                     validation_data = valGenRev,
                                     validation_steps = valSteps)
lossBidirec = history.history['loss']
valLossBidirec = history.history['val_loss']
epochs = range(len(lossBidirec))
import matplotlib.pyplot as plt
plt.figure()
plt.plot(epochs,lossBidirec,'bo',label = 'Training Loss')
plt.plot(epochs,val_lossBidirec,'b',label = 'Validation Loss')
plt.title("Training and Validation losses")
plt.legend()
plt.show()








