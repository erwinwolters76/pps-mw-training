from pathlib import Path
from typing import Dict, Optional
import json

from xarray import DataArray, Dataset  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from pps_mw_training.models.predictors.mlp_predictor import MlpPredictor


def evaluate_model(
    model: MlpPredictor,
    dataset: Dataset,
    missing_fraction: float,
    output_path: Path,
) -> None:
    """Evaluate model."""
    plot_fit_history(output_path)
    for param in model.input_params:
        if dataset[param].dtype == np.float32:
            filt = np.random.rand(dataset[param].size) < missing_fraction
            dataset[param].values[filt] = np.nan
    predicted = model.predict(dataset)
    evaluate_quantile_performance(dataset, predicted, output_path)
    evaluate_distribution_performance(dataset, predicted, output_path)
    plot_prediction(dataset, predicted, output_path)


def plot_fit_history(
    history_path: Path
) -> None:
    """Plot training fit history."""
    with open(history_path / "fit_history.json") as history_file:
        history = json.load(history_file)
    plt.figure()
    plt.plot(history["loss"], label="training")
    plt.plot(history["val_loss"], label="validation")
    plt.ylabel("loss")
    plt.ylim(
        [0, np.max([np.column_stack([history["loss"], history["val_loss"]])])]
    )
    plt.legend()
    plt.savefig(history_path / "fit_history.png")


def evaluate_quantile_performance(
    true_state: Dataset,
    predicted_state: Dataset,
    output_path: Path,
) -> None:
    """Evaluate quantile performance."""
    stats: Dict[str, Dict[float, float]] = {}
    for param in predicted_state:
        stats[param] = {}
        for j, quantile in enumerate(predicted_state["quantile"].values):
            obtained_quantile = np.count_nonzero(
                predicted_state[param][:, j].values > true_state[param].values
            ) / predicted_state.t.size
            stats[param][float(quantile)] = obtained_quantile
    with open(output_path / "quantile_stats.json", "w") as outfile:
        outfile.write(json.dumps(stats, indent=4))


def evaluate_distribution_performance(
    true_state: Dataset,
    predicted_state: Dataset,
    output_path: Path,
) -> None:
    """Evaluate retrieved distribution performance."""
    min_value = 1e-5
    min_value_plot = 1e-4
    max_value = 10
    n_points = 100
    quantile = 0.5
    plot_range = np.array([min_value_plot, max_value])
    edges = np.logspace(np.log10(min_value), np.log10(max_value), n_points)
    true_stats = get_distribution(
        true_state["IWP"] + min_value,
        edges,
        predicted_state["quantile"].values,
    )
    predicted_stats = get_distribution(
        predicted_state["IWP"].sel(quantile=quantile) + min_value,
        edges,
        predicted_state["quantile"].values,
    )
    stats = {
        "true": true_stats.data.values.tolist(),
        "predicted": predicted_stats.data.values.tolist(),
        "quantile": true_stats["quantile"].values.tolist(),

        "cvm":  np.sqrt(  # Cramer-von Mises criterion
            np.trapz(
                (true_stats.dist - predicted_stats.dist) ** 2,
                predicted_stats.dist
            )
        )
    }
    with open(output_path / "prediction_dist_stats.json", "w") as outfile:
        outfile.write(json.dumps(stats, indent=4))
    plt.figure(figsize=(7, 10))
    plt.subplot(2, 1, 1)
    plt.loglog(true_stats.x, true_stats.pdf, label="True")
    plt.loglog(predicted_stats.x, predicted_stats.pdf, label="Predicted")
    plt.xlabel("IWP [kg/m2]")
    plt.ylabel("PDF [1/(kg/m2)]")
    plt.xlim(plot_range)
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.semilogx(true_stats.x, true_stats.dist)
    plt.semilogx(predicted_stats.x, predicted_stats.dist)
    plt.xlabel("IWP [kg/m2]")
    plt.ylabel("CDF [-]")
    plt.xlim(plot_range)
    plt.ylim([-0.03, 1.03])
    plt.grid(True)
    plt.savefig(output_path / "prediction_distribution_performance.png")
    plt.figure(figsize=(7, 5))
    plt.semilogy(true_stats["quantile"], true_stats.data, label="True")
    plt.semilogy(
        predicted_stats["quantile"], predicted_stats.data, label="Predicted",
    )
    plt.xlabel("quantile [-]")
    plt.ylabel("IWP [kg/m2]")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path / "prediction_quantile_performance.png")


def get_distribution(
    data: DataArray,
    edges: np.ndarray,
    quantiles: np.ndarray,
) -> Dataset:
    """Get pdf and distribution."""
    lower = edges[0:-1]
    upper = edges[1::]
    center = (lower + upper) / 2.
    bin_size = upper - lower
    count, _ = np.histogram(data, edges)
    pdf = count / bin_size / np.sum(count)
    dist = np.cumsum(pdf * bin_size)
    return Dataset(
        data_vars={
            "pdf": ("x", pdf),
            "dist": ("x", dist),
            "data": ("quantile", np.interp(quantiles, dist, center))
        },
        coords={
            "x": ("x", center),
            "quantile": quantiles,
        },
        attrs={
            "mean": np.trapz(center * pdf, center),
        }
    )


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
    output_path: Path,
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
    plt.savefig(output_path / "prediction_performance.png")
