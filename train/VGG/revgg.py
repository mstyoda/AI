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
import os
import sys

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
train = pd.read_csv('train.csv')
labels = train.ix[:,0].values.astype('int32')

x_train = (train.ix[:,1:].values).astype('float32')

x_train /= 255.0
y_train = np_utils.to_categorical(labels)

(xtr, ytr), (xte, yte) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    xtr = xtr.reshape(xtr.shape[0], 1, img_rows, img_cols)
    xte = xte.reshape(xte.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    xtr = xtr.reshape(xtr.shape[0], img_rows, img_cols, 1)
    xte = xte.reshape(xte.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

ytr = keras.utils.to_categorical(ytr, num_classes)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
(x_train,x_test) = (x_train[0:35000],x_train[35000:])
(y_train,y_test) = (y_train[0:35000],y_train[35000:])

model = load_model("0.968vgg.h5")

best = 0.968
for i in range(0,30):
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs = 1,
            verbose=1,
            validation_data=(x_test, y_test))

  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  if (score[1] > best):
    best = score[1]
    filepath = str(score[1]) + 'vgg.h5' 
    model.save(filepath)