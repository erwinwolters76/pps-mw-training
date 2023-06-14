"""model.py
This module provides an implementation of a quantile regression
neural network. The quantile loss function handles both multiple
quantiles and parameters. 
"""
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.data import Dataset

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from pps_mw_training import ici
from pps_mw_training.utils import (
    create_simple_training_dataset,
    split_dataset,
    to_numpy_arrays,
)


# model parameters
QUANTILES = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
N_CHANNELS = 4
N_PARAMS = 2
N_HIDDEN_LAYERS = 4
N_NEURONS = 128
ACTIVATION = "relu"
# training parameters
BATCH_SIZE = 4096
EPOCHS = 20
INITIAL_LEARNING_RATE = 0.0001
FIRST_DECAY_STEPS = 1000
T_MUL = 2.0
M_MUL = 1.0
ALPHA = 0.0
# training dataset parameters
N_TRAINING_SAMPLES = 1000000
TRAIN_FRACTION = 0.7
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
MODEL_WEIGHTS = Path("saved_model/pretrained_weights.h5")


def quantile_loss(
    n_params: int,
    quantiles: List[float],
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) ->  tf.Tensor:
    """Quantile loss function."""
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
        loss=lambda y, y_p: quantile_loss(n_params, quantiles, y, y_p),
        metrics=['accuracy'],
    )
    return model


def evaluate_model(
    model: keras.Sequential,
    dataset: Dataset,
    n_params: int,
    quantiles: List[float],
) -> None:
    """Evaluate model."""
    score = model.evaluate(dataset, verbose=0)
    print(f"Test loss: {score[0]}")
    print(f"Test accuracy: {score[1]}")
    observation, state = to_numpy_arrays(dataset.unbatch())
    predicted = model(observation)
    evaluate_quantile_performance(
        state,
        predicted,
        n_params,
        quantiles,
    )
    plot_prediction(
        state,
        predicted,
        n_params,
        quantiles,
    )


def evaluate_quantile_performance(
    true_state: np.ndarray,
    predicted_state: np.ndarray,
    n_params: int,
    quantiles: List[float],
) -> None:
    """Evaluate quantile performance."""
    print("Evaluate quantile performance")
    for i in range(n_params):
        for j in range(len(quantiles)):
            obtained_quantile = np.count_nonzero(
                predicted_state[:, i * len(quantiles) + j] > true_state[:, i],
            ) / predicted_state.shape[0]
            print(f"param{i} quantile {quantiles[j]}: {obtained_quantile}")


def plot_prediction(
    true_state: np.ndarray,
    predicted_state: np.ndarray,
    n_params: int,
    quantiles: List[float],
    plot_error_bar: bool = True,
) -> None:
    """Plot prediction."""
    n_quantiles = len(quantiles)
    for i in range(n_params):
        plt.subplot(3, 2, i + 1)
        predicted = predicted_state[:, int(n_quantiles // 2 + i * n_quantiles)]
        value_range = [
            np.floor(np.min(true_state[:, i])),
            np.ceil(np.max(true_state[:, i]))
        ]
        plt.plot(value_range, value_range, "-k", label="1-to-1")
        if plot_error_bar:
            plt.errorbar(
                true_state[:, i],
                predicted,
                [
                    np.abs(predicted - predicted_state[:, i * n_quantiles]),
                    np.abs(predicted_state[:, (i + 1) * (n_quantiles - 1)] - predicted),
                ],
                fmt=f"C{i}.",
                label=f"median param{i}",
                errorevery=10,
            )
        else:
            plt.plot(
                true_state[:, i],
                predicted,
                f"C{i}.",
                label=f"median param{i}",
            )
        plt.grid(True)
    plt.legend()
    plt.xlim(value_range)
    plt.ylim(value_range)
    plt.xlabel("true state [-]")
    plt.ylabel("predicted state [-]")
    plt.show()


if __name__ == "__main__":
    # Create, train, and evaluate model
    do_ici_model = True
    train = True
    if do_ici_model:
        N_PARAMS = len(ici.STATE_PARAMS)
        N_CHANNELS = len(ici.CHANNEL_PARAMS)
        full_dataset = ici.load_retrieval_database()
    else:
        full_dataset = create_simple_training_dataset(
            N_TRAINING_SAMPLES,
            N_CHANNELS,
            N_PARAMS,
        )
    training_dataset, test_dataset, validation_dataset = split_dataset(
        full_dataset,
        TRAIN_FRACTION,
        VALIDATION_FRACTION,
        TEST_FRACTION,
    )
    model = create_model(
        N_CHANNELS,
        N_HIDDEN_LAYERS,
        N_NEURONS,
        ACTIVATION,
        N_PARAMS,
        QUANTILES,
    )
    if train:
        model.fit(
            training_dataset.batch(batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            verbose=1,
            validation_data=validation_dataset.batch(batch_size=BATCH_SIZE),
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    MODEL_WEIGHTS,
                    save_best_only=True,
                    save_weights_only=True,
                )
            ],
        )
    model.load_weights(MODEL_WEIGHTS)
    evaluate_model(
        model,
        test_dataset.batch(batch_size=BATCH_SIZE),
        N_PARAMS,
        QUANTILES,
    )
