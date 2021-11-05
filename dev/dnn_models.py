import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization


def test_model_1(input_dim, label):
    inputs = Input(shape=(input_dim,), name=label + "_input")
    x = Dense(128, name=label + "_layer_1", activation="tanh")(inputs)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(64, name=label + "_layer_2", activation="tanh")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(32, name=label + "_layer_3", activation="tanh")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    outputs = Dense(1, name=label + "_output", activation="sigmoid")(x)
    dnn = Model(inputs=inputs, outputs=outputs)
    return dnn


def test_model_2(input_dim, label):
    inputs = Input(shape=(input_dim,), name=label + "_input")
    x = Dense(2, name=label + "_layer_1", activation="tanh")(inputs)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(2, name=label + "_layer_2", activation="tanh")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(2, name=label + "_layer_3", activation="tanh")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    outputs = Dense(1, name=label + "_output", activation="sigmoid")(x)
    dnn = Model(inputs=inputs, outputs=outputs)
    return dnn
