import numpy as np 
import pandas as pd 
import sklearn
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation


def build_model(X_train, y_train):
    layer_1 = 400
    layer_2 = 200

    model = Sequential()
    model.add(Dense(input_dim=train.shape[1], 
                    output_dim=layer_1, 
                    init='uniform',
                    activation='tanh'))

    model.add(Dense(input_dim=layer_1,
                    output_dim=layer_2,
                    init='uniform',
                    activation='tanh'))

    model.add(Dense(input_dim=layer_2,
                    output_dim=1,
                    init='uniform',
                    activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])
	
	model.fit(X_train, y_train, nb_epoch=30, batch_size=50)
    return model




