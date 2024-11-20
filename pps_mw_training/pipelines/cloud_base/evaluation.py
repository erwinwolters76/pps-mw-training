from pathlib import Path
import json

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow import keras
import xarray as xr  # type: ignore
from pps_mw_training.pipelines.cloud_base import settings
from pps_mw_training.models.predictors.unet_predictor import UnetPredictor
from pps_mw_training.utils.scaler import get_scaler


def evaluate_model(
    predictor: UnetPredictor,
    test_data: tf.data.Dataset,
    output_path: Path,
) -> None:
    """Evaluate model, very rudimentary for the moment."""
    predictor.model.summary()
    keras.utils.plot_model(
        predictor.model.build_graph(
            settings.IMAGE_SIZE, len(settings.INPUT_PARAMS)
        ),
        to_file=settings.MODEL_CONFIG_PATH / "model.png",
        show_shapes=True,
        show_dtype=False,
        show_layer_names=False,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
        layer_range=None,
        show_layer_activations=True,
    )
    stats = get_stats(predictor, test_data, output_path)
    with open(output_path / "retrieval_stats.json", "w") as outfile:
        outfile.write(json.dumps(stats, indent=4))


def get_stats(
    predictor: UnetPredictor, test_data: tf.data.Dataset, output_path: Path
) -> dict[str, float]:
    """Get stats."""
    preds = []
    labels = []
    inputs = []

    for input_data, label_data in test_data:
        for p in predictor.model(input_data).numpy():
            preds.append(p)
        for r in label_data.numpy():
            labels.append(r)
        for i in input_data.numpy():
            inputs.append(i)

    preds_all = np.array(preds)
    labels_all = np.array(labels).squeeze()
    inputs_all = np.array(inputs)

    model_config_file = output_path / "network_config.json"
    with open(model_config_file) as config_file:
        config = json.load(config_file)
        n_outputs = len(config["quantiles"])
        input_params = config["input_parameters"]

    imedian = n_outputs // 2
    label_scaler = get_scaler(settings.LABEL_PARAMS)
    labels_all = label_scaler.reverse(labels_all, 0)
    preds_all = label_scaler.reverse(preds_all, 0)
    fill_value_mask = inputs_all == config["fill_value"]
    input_scaler = get_scaler(input_params)
    inputs_all = np.stack(
        [
            input_scaler.reverse(inputs_all[:, :, :, idx], idx)
            for idx in range(len(input_params))
        ],
        axis=3,
    )
    inputs_all[fill_value_mask] = np.nan

    plot_preds(
        labels_all,
        preds_all[:, :, :, imedian],
    )
    write_netcdf(
        output_path / "predictions.nc", labels_all, inputs_all, preds_all
    )

    mask = labels_all > 0
    return {
        "rmse": float(
            np.sqrt(
                np.nanmean(
                    (preds_all[:, :, :, imedian][mask] - labels_all[mask]) ** 2
                )
            )
        ),
        "mae": float(
            np.mean(
                np.abs(preds_all[:, :, :, imedian][mask] - labels_all[mask])
            )
        ),
        "bias": float(
            np.mean(preds_all[:, :, :, imedian][mask] - labels_all[mask])
        ),
    }


def write_netcdf(
    ncfile: Path, labels: np.ndarray, inputs: np.ndarray, preds: np.ndarray
):
    """write output to netcdf file"""

    nimages = labels.shape[0]
    x = labels.shape[1]
    y = labels.shape[2]
    ds = xr.Dataset(
        {
            "labels": (["nimages", "x", "y"], labels),
            "inputs": (["nimages", "x", "y", "ninputs"], inputs),
            "preds": (["nimages", "x", "y", "nquantiles"], preds),
        },
        coords={
            "nimages": np.arange(nimages),
            "x": np.arange(x),
            "y": np.arange(y),
            "ninputs": [item["name"] for item in settings.INPUT_PARAMS],
            "nquantiles": settings.QUANTILES,
        },
    )
    ds.to_netcdf(
        ncfile,
        mode="w",
    )


def plot_preds(labels: np.ndarray, preds: np.ndarray):
    fig, ax = plt.subplots(1, 2, figsize=[8, 4])
    ax = ax.ravel()
    valid_labels = labels > 0
    ax[0].scatter(labels[valid_labels], preds[valid_labels])
    ax[0].set_xlabel("Labels")
    ax[1].set_ylabel("Preds")
    ax[1].hist(
        labels[valid_labels],
        bins=100,
        density=True,
        histtype="step",
        label="Labels",
    )
    ax[1].hist(
        preds[valid_labels],
        bins=100,
        density=True,
        histtype="step",
        label="Preds",
    )
    ax[1].legend()
    ax[1].set_xlabel("labels/preds")
    ax[1].set_ylabel("Distribution")
    plt.show()
    fig.tight_layout()
