from typing import Optional
from tensorflow import keras  # type: ignore
from xarray import Dataset  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore


def evaluate_model(
    model: keras.Sequential,
    dataset: Dataset,
) -> None:
    """Evaluate model."""
    predicted = model.predict(dataset)
    evaluate_quantile_performance(dataset, predicted)
    plot_prediction(dataset, predicted)


def evaluate_quantile_performance(
    true_state: Dataset,
    predicted_state: Dataset,
) -> None:
    """Evaluate quantile performance."""
    for param in predicted_state:
        for j, quantile in enumerate(predicted_state["quantile"].values):
            obtained_quantile = np.count_nonzero(
                predicted_state[param][:, j].values > true_state[param].values
            ) / predicted_state.t.size
            print(f"{param} quantile {quantile}: {obtained_quantile}")


def get_stats(
    bins,
    true_state: np.ndarray,
    predicted_state: np.ndarray,
    extra_filter: Optional[np.ndarray] = None,
):
    retrived_median = np.zeros(bins.size - 1)
    for idx in range(bins.size - 1):
        filt = (true_state >= bins[idx]) & (true_state < bins[idx + 1])
        if extra_filter is not None:
            filt = filt & extra_filter
        retrived_median[idx] = np.median(predicted_state[filt])
    return (bins[0:-1] + bins[1::]) / 2., retrived_median


def plot_prediction(
    true_state: Dataset,
    predicted_state: Dataset,
    n_edges: int = 100,
    min_iwp: float = 0.01,
) -> None:
    """Plot prediction and the known state."""
    plt.figure(figsize=(12, 10))
    n_quantiles = predicted_state["quantile"].size
    n_edges = 200
    for i, param in enumerate(predicted_state):
        plt.subplot(3, 2, i + 1)
        idx_q = int(n_quantiles // 2)
        if param in ["IWP", "LWP", "RWP"]:
            min_value = 1e-4
            max_value = 10
            value_range = np.array([min_value, max_value])
            plt.loglog(value_range, value_range, "-k", label="1-to-1")
            bins = np.logspace(
                np.log10(min_value),
                np.log10(max_value),
                n_edges,
            )
            for idx in [-1, 0, 1]:
                center, q = get_stats(
                    bins,
                    true_state[param].values,
                    predicted_state[param][:, idx_q + idx].values,
                )
                plt.loglog(center, q, "-", label=f"Q{idx + 2}")
        else:
            scale = 1e6 if param == "Dmean" else 1.
            min_value = np.min(true_state[param]) * scale
            max_value = np.max(true_state[param]) * scale
            value_range = np.array([min_value, max_value])
            plt.plot(value_range, value_range, "-k", label="1-to-1")
            bins = np.linspace(min_value, max_value, n_edges)
            if param in ["Dmean", "Zmean"]:
                extra_filter = (
                    (true_state["IWP"].values >= min_iwp)
                    & (predicted_state["IWP"][:, idx_q].values >= min_iwp)
                )
            else:
                extra_filter = None
            for idx in [-1, 0, 1]:
                center, q = get_stats(
                    bins,
                    true_state[param].values * scale,
                    predicted_state[param][:, idx_q + idx].values * scale,
                    extra_filter=extra_filter,
                )
                plt.plot(center, q, "-", label=f"Q{idx + 2}")
        plt.xlim(value_range)
        plt.ylim(value_range)
        plt.title(param)
        plt.grid(True)
        plt.legend()
        if i >= 4:
            plt.xlabel("true state [-]")
        if i in [0, 2, 4]:
            plt.ylabel("predicted state [-]")
    plt.show()
