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


def plot_prediction(
    true_state: np.ndarray,
    predicted_state: np.ndarray,
    plot_error_bar: bool = False,
) -> None:
    """Plot prediction and the known state."""
    n_quantiles = predicted_state["quantile"].size
    for i, param in enumerate(predicted_state):
        plt.subplot(3, 2, i + 1)
        predicted = predicted_state[param][:, int(n_quantiles // 2)]
        value_range = np.array(
            [np.min(true_state[param]), np.max(true_state[param])]
        )
        if param in ["IWP", "LWP", "RWP"]:
            plt.loglog(value_range, value_range, "-k", label="1-to-1")
            plt.loglog(true_state[param], predicted, ".")
            plt.xlim(1e-6, 30)
            plt.ylim(1e-6, 30)
        else:
            scale = 1e6 if param == "Dmean" else 1.
            value_range *= scale  
            plt.plot(value_range, value_range, "-k", label="1-to-1")
            plt.plot(scale * true_state[param], scale * predicted, ".")
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
