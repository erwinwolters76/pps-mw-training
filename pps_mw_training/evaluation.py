from pathlib import Path
from typing import List, Tuple

from tensorflow import keras
from tensorflow.data import Dataset

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from pps_mw_training import ici
from pps_mw_training.utils import to_numpy_arrays


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
                predicted_state[:, i * len(quantiles) + j]
                > true_state[:, i],
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
