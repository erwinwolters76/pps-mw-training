from typing import List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.data import Dataset

import numpy as np  # type: ignore


# learning rate parameters
INITIAL_LEARNING_RATE = 0.0001
FIRST_DECAY_STEPS = 1000
T_MUL = 2.0
M_MUL = 1.0
ALPHA = 0.0


def quantile_loss(
    n_params: int,
    quantiles: List[float],
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) ->  tf.Tensor:
    """Quantile loss function handling multiple quantiles and parameters."""
    s = len(quantiles)
    q = tf.constant(np.tile(quantiles, n_params), dtype=tf.float32)
    e = tf.concat(
        [
            tf.expand_dims(y_true[:, i], 1) - y_pred[:, i * s: (i + 1) * s]
            for i in range(n_params) 
        ],
        axis=1
    ) 
    v = tf.maximum(q * e, (q - 1) * e)
    return tf.reduce_mean(v)


def create_model(
    n_channels: int,
    n_hidden_layers: int,
    n_neurons: int,
    activation: str,
    n_params: int,
    quantiles: List[float],
) -> keras.Sequential:
    """Create the model."""
    model = keras.Sequential()
    model.add(keras.Input(shape=(n_channels,)))
    for _ in range(n_hidden_layers):
        model.add(layers.Dense(n_neurons, activation=activation))
    model.add(layers.Dense(n_params * len(quantiles), activation="linear"))
    model.summary()
    learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        first_decay_steps=FIRST_DECAY_STEPS,
        t_mul=T_MUL,
        m_mul=M_MUL,
        alpha=ALPHA,
        name=None,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=lambda y_true, y_pred: quantile_loss(n_params, quantiles, y_true, y_pred),
        metrics=['accuracy'],
    )
    return model
