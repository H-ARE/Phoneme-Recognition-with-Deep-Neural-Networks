import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D,Conv2D,Conv2DTranspose,BatchNormalization
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers, initializers
from sklearn.metrics import confusion_matrix
from lab3_proto import *

def modelcreator():
    """
    Defines the neural network model
    """
    model=Sequential()
    model.add(Dense(256,input_shape=[40],activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(61,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


def train(data):
    """
    Stochastic gradient descent.
    """
    model=modelcreator()
    model.summary()
    epochs=3
    batch_size=256

    x_train=alldata['mspec_train_x']
    y_train=alldata['train_y']

    x_test=alldata['mspec_test_x']
    y_test=alldata['test_y']

    yt=np.zeros([y_train.shape[0],61])
    yte=np.zeros([y_test.shape[0],61])

    for i in range(y_train.shape[0]):
        yt[i,int(y_train[i])]=1

    for i in range(y_test.shape[0]):
        yte[i,int(y_test[i])]=1

    y_train=yt
    y_test=yte
    a=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.1)
    print(a.history.keys())
    print(model.evaluate(x_test,y_test))
    return a,model



