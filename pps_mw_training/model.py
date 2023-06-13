"""model.py
This module provides an implementation of a quantile regression
neural network. The quantile loss function handles both multiple
quantiles and parameters. 
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.data import Dataset
from typing import List, Tuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore


# model parameters
QUANTILES = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
N_CHANNELS = 4
N_PARAMS = 2
N_HIDDEN_LAYERS = 2
N_NEURONS = 128
ACTIVATION = "relu"
# training parameters
BATCH_SIZE = 128
EPOCHS = 20
# training dataset parameters
N_TRAINING_SAMPLES = 1000000
TRAIN_FRACTION = 0.7
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


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


def build_model(
    n_channels: int,
    n_hidden_layers: int,
    n_neurons: int,
    activation: str,
    n_params: int,
    quantiles: List[float],
) -> keras.Sequential:
    """Build the quantile model."""
    model = keras.Sequential()
    model.add(keras.Input(shape=(n_channels,)))
    for _ in range(n_hidden_layers):
        model.add(layers.Dense(n_neurons, activation=activation))
    model.add(layers.Dense(n_params * len(quantiles), activation="linear"))
    model.summary()
    learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.0001,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.0,
        name=None,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=lambda y, y_p: quantile_loss(n_params, quantiles, y, y_p),
        metrics=['accuracy'],
    )
    return model


def create_simple_training_dataset(
    n_samples: int,
    n_channels: int,
    n_params: int,
) -> Dataset:
    """Create a simple 'toy' training dataset."""
    state = np.stack(
        [i + 0.5 * np.random.randn(n_samples) for i in range(n_params)]
    ).T
    observation = np.zeros((n_samples, n_channels))
    for i in range(n_channels):
        noise = 0.01 * np.random.randn(n_samples)
        weights = np.random.rand(n_params)
        weighted = np.matmul(state, weights / np.sum(weights))
        observation[:, i] = weighted + np.sin(weighted) + noise
    return Dataset.from_tensor_slices((observation, state))


def split_dataset(
    dataset: Dataset
) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into a training, validation, and test dataset."""
    n_samples = len(dataset)
    train_size = int(TRAIN_FRACTION * n_samples)
    val_size = int(VALIDATION_FRACTION * n_samples)
    test_size = int(TEST_FRACTION * n_samples)
    training_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    validation_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)
    return training_dataset, test_dataset, validation_dataset


def to_numpy_arrays(
    dataset: Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Extract observation and state arrays from dataset."""
    observation, state = tuple(zip(*dataset))
    return np.array(observation), np.array(state)


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
) -> None:
    """Plot prediction."""
    n_quantiles = len(quantiles)
    value_range = [
        np.floor(np.min(predicted_state)),
        np.ceil(np.max(predicted_state))
    ]
    plt.plot(value_range, value_range, "-k", label="1-to-1")
    for i in range(n_params):
        predicted = predicted_state[:, int(n_quantiles // 2 + i * n_quantiles)]
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
    plt.legend()
    plt.xlim(value_range)
    plt.ylim(value_range)
    plt.xlabel("true state [-]")
    plt.ylabel("predicted state [-]")
    plt.show()


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


if __name__ == "__main__":
    # Build fit and evaluate model
    model = build_model(
        N_CHANNELS,
        N_HIDDEN_LAYERS,
        N_NEURONS,
        ACTIVATION,
        N_PARAMS,
        QUANTILES,
    )
    full_dataset = create_simple_training_dataset(
        N_TRAINING_SAMPLES,
        N_CHANNELS,
        N_PARAMS,
    )
    training_dataset, test_dataset, validation_dataset = split_dataset(
        full_dataset
    )
    model.fit(
        training_dataset.batch(batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        verbose=1,
        validation_data=validation_dataset.batch(batch_size=BATCH_SIZE),
    )
    evaluate_model(
        model,
        test_dataset.batch(batch_size=BATCH_SIZE),
        N_PARAMS,
        QUANTILES,
    )
