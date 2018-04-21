# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 22:38:53 2018

@author: jibin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 12:51:14 2018

@author: jibin
"""

# Import libraries and modules
import numpy as np

 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

 
# Preprocessing
X_train = X_train.reshape(X_train.shape[0], 200,200,1) 
X_test = X_test.reshape(X_test.shape[0], 200,200,1) 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255




Y_train = np_utils.to_categorical(Y_train, 120)
Y_test = np_utils.to_categorical(Y_test, 120)
 

# Setting up the model
model = Sequential()
 
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(200,200,1))) #Changing for tensorflow
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))t
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(120, activation='softmax'))
 
# Compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# Fit model on training data
model.fit(X_train, Y_train, 
          batch_size=32, epochs=1, verbose=1)
 
# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=1)

