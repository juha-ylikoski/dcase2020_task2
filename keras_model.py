"""
 @file   keras_model.py
 @brief  Script for keras model definition
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
# from import
import numpy as np
import keras.models
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation, Flatten,Reshape

########################################################################
# keras model
########################################################################
def get_model(inputDim, x):
    """
    define the keras model
    the model based on the simple dense auto encoder 
    (128*128*128*128*8*128*128*128*128)
    """
    inputLayer = Input(shape=inputDim)
    print(inputLayer.shape)
    h = Flatten()(inputLayer)
    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    # h = Dense(128)(h)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)
    
    h = Dense(8)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    # h = Dense(128)(h)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(np.prod(inputDim))(h)
    h = Reshape(inputDim)(h)
    return Model(inputs=inputLayer, outputs=h)
#########################################################################


def load_model(file_path):
    return keras.models.load_model(file_path)


if __name__ == '__main__':
    m = get_model((64,64,1))
    m.summary()