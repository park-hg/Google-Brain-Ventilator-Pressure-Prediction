import gc
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Permute, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.layers import Concatenate, LSTM, GRU
from tensorflow.keras.layers import Bidirectional, Multiply, Softmax, Add, LayerNormalization
from tensorflow.keras.layers import MaxPool1D, AveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecayRestarts

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K



def ConvBlock1(filters):
    return keras.Sequential(layers=[
        Conv1D(filters=filters, kernel_size=5, padding='same', activation='selu'),
        Conv1D(filters=filters, kernel_size=3, padding='same', activation='selu'),
    ])
    
def ConvBlock2(filters):
    return keras.Sequential(layers=[
        MaxPool1D(strides=1, padding='same'),
        Conv1D(filters=filters, kernel_size=5, padding='same', activation='selu'),
        Conv1D(filters=filters, kernel_size=3, padding='same', activation='selu'),
    ])

def ConvBlock3(filters):
    return keras.Sequential(layers=[
        Conv1D(filters=filters, kernel_size=5, padding='same', activation='selu'),
        Conv1D(filters=filters, kernel_size=3, padding='same', activation='selu'),
        AveragePooling1D(strides=1, padding='same'),
    ])

def get_model():
    x_input = Input(shape=(80, 64))
    
    x1 = Bidirectional(LSTM(units=768, return_sequences=True))(x_input)
    x2 = Bidirectional(LSTM(units=512, return_sequences=True))(x1)
    x3 = Bidirectional(LSTM(units=384, return_sequences=True))(x2)
    x4 = Bidirectional(LSTM(units=256, return_sequences=True))(x3)
    x5 = Bidirectional(LSTM(units=128, return_sequences=True))(x4)
    
    z2 = ConvBlock1(768)(x2)

    z31 = Multiply()([x3, z2])
    z31 = BatchNormalization()(z31)
    z3 = ConvBlock2(512)(z31)

    z41 = Multiply()([x4, z3])
    z41 = BatchNormalization()(z41)
    z4 = ConvBlock2(256)(z41)

    z51 = Multiply()([x5, z4])
    z51 = BatchNormalization()(z51)
    z5 = ConvBlock3(128)(z51)

    x = Concatenate(axis=2)([x5, z2, z3, z4, z5])
    x = Dense(units=128, activation='selu')(x)
    x_output = Dense(units=1)(x)

    model = keras.models.Model(inputs=x_input, outputs=x_output, 
                  name='lstm_cnn')
    return model