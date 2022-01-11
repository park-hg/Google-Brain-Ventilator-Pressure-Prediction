import gc
import numpy as np
import pandas as pd

from utils import *
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

from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, GroupKFold, KFold

from IPython.display import display


np.random.seed(42)
tf.random.set_seed(42)

train = pd.read_pickle('../input/ventilator-pressure-prediction/train_preprocessed.pkl')
train = reduce_mem_usage(train)

targets = train['pressure'].to_numpy().reshape(-1, 80)
u_outs = train['u_out'].to_numpy().reshape(-1, 80)
train = train.drop(['pressure','id', 'breath_id','one','count',
                    'breath_id_lag','breath_id_lag2','breath_id_lagsame',
                    'breath_id_lag2same'], axis=1,)

RS = RobustScaler()
train = RS.fit_transform(train)
train = train.reshape(-1, 80, train.shape[-1])


def masked_l1_loss(y_true, y_pred, cols=80):
    u_out = tf.reshape(y_true[:, cols: ], [-1])
    y = tf.reshape(y_true[:, :cols ], [-1])
    y_pred = tf.reshape(y_pred, [-1])
    w = 1 - u_out
    mae = w * tf.abs(y - y_pred)
    return tf.reduce_sum(mae, axis=-1) / tf.reduce_sum(w, axis=-1)

EPOCH = 400
BATCH_SIZE = 512
NUM_FOLDS = 10

VERSION = 'cnn'
if VERSION == 'cnn':
    import model_cnn
    model = model_cnn.get_model()
elif VERSION == 'dense':
    import model_mlp
    model = model_mlp.get_model()


# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
# instantiate a distribution strategy
tpu_strategy = tf.distribute.TPUStrategy(tpu)
with tpu_strategy.scope():
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
    test_preds = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        print('-'*15, '>', f'Fold {fold}', '<', '-'*15)
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]
        u_out_train, u_out_valid = u_outs[train_idx], u_outs[test_idx]

        model.compile(optimizer="adam", loss=masked_l1_loss)
        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.75, patience=10, verbose=1)
        es = EarlyStopping(monitor="val_loss", patience=35, verbose=1, mode="min", restore_best_weights=True)

        chpt_path = VERSION + f"/folds{fold}.h5"
        sv = keras.callbacks.ModelCheckpoint(
            chpt_path, monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=False, mode='auto', save_freq='epoch',
            options=None)

        model.fit(X_train, np.append(y_train, u_out_train, axis =1),
                validation_data=(X_valid, np.append(y_valid, u_out_valid, axis =1)),
                epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[lr, sv, es])