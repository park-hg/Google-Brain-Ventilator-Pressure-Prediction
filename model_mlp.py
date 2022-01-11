import gc
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Permute, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Concatenate, LSTM, GRU
from tensorflow.keras.layers import Bidirectional, Multiply, Softmax, Add, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecayRestarts

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

def get_model():
    
    x_input = Input(shape=(80, 64))
    
    x1 = Bidirectional(LSTM(units=768, return_sequences=True))(x_input)
    x2 = Bidirectional(LSTM(units=512, return_sequences=True))(x1)
    x3 = Bidirectional(LSTM(units=384, return_sequences=True))(x2)
    x4 = Bidirectional(LSTM(units=256, return_sequences=True))(x3)
    x5 = Bidirectional(LSTM(units=128, return_sequences=True))(x4)
    
    z2 = Dense(1024, activation='selu')(x2)
    z2 = Dense(896, activation='selu')(z2)
    z2 = Dense(768, activation='selu')(z2)

    z31 = Multiply()([x3, z2])
    z31 = BatchNormalization()(z31)
    z3 = Dense(896, activation='selu')(z31)
    z3 = Dense(768, activation='selu')(z3)
    z3 = Dense(512, activation='selu')(z3)

    z41 = Multiply()([x4, z3])
    z41 = BatchNormalization()(z41)
    z4 = Dense(768, activation='selu')(z41)
    z4 = Dense(512, activation='selu')(z4)
    z4 = Dense(256, activation='selu')(z4)

    z51 = Multiply()([x5, z4])
    z51 = BatchNormalization()(z51)
    z5 = Dense(512, activation='selu')(z51)
    z5 = Dense(256, activation='selu')(z5)
    z5 = Dense(128, activation='selu')(z5)

    x = Concatenate(axis=2)([x5, z2, z3, z4, z5])
    x = Dense(units=128, activation='selu')(x)
    x_output = Dense(units=1)(x)

    model = keras.models.Model(inputs=x_input, outputs=x_output, 
                  name='lstm_mlp')
    return model