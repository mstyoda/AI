from __future__ import print_function
import keras
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pandas as pd
import numpy as np
import copy

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (28, 28, 1)
# the data, shuffled and split between train and test sets

def getData():
  train = pd.read_csv('train.csv')
  labels = train.ix[:,0].values.astype('int32')
  X = (train.ix[:,1:].values).astype('float32')
  X /= 255.0
  X = X.reshape(X.shape[0], img_rows, img_cols, 1)
  md = load_model("0.995916666667vgg3.h5")
  Y = md.predict_classes(X, verbose=0)
  print (labels,Y)
  cnt = 0
  for i in range(0,Y.shape[0]):
    if Y[i] != labels[i]:
      cnt += 1
  print (cnt)

  x_train = copy.deepcopy(X[0:len(X)-cnt])
  y_train = copy.deepcopy(Y[0:len(Y)-cnt])

  j = 0
  for i in range(0,Y.shape[0]):
    if Y[i] == labels[i]:
      x_train[j] = X[j]
      y_train[j] = Y[j]
      j += 1
  
  (x_test, y_test), (xtr, ytr) = mnist.load_data()
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  x_test = x_test.astype('float32')
  x_test /= 255
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test = getData()

model = Sequential()
model.add(Conv2D(20, (4, 4), activation='relu', input_shape= input_shape))
model.add(Conv2D(20, (3, 3), activation='relu', input_shape= input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
filepath = str(score[1]) + 'vgg4.h5' 
model.save(filepath)