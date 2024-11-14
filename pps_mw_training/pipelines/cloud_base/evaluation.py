from pathlib import Path
import json

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
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
    plot_preds(predictor, test_data)


def get_stats(
    predictor: UnetPredictor, test_data: tf.data.Dataset, output_path: Path
) -> dict[str, float]:
    """Get stats."""
    preds = []
    labels = []
    images = []

    for input_data, label_data in test_data:
        for p in predictor.model(input_data).numpy():
            preds.append(p)
        for r in label_data.numpy():
            labels.append(r)
        for i in input_data.numpy():
            images.append(i)

    preds_all = np.array(preds)
    labels_all = np.array(labels)
    images_all = np.array(images)

    model_config_file = output_path / "network_config.json"
    with open(model_config_file) as config_file:
        config = json.load(config_file)
        n_outputs = len(config["quantiles"])

    label_scaler = get_scaler(settings.LABEL_PARAMS)
    labels_all = label_scaler.reverse(labels_all, 0)
    preds_all = label_scaler.reverse(preds_all, 0)
    write_netcdf(
        output_path / "predictions.nc", labels_all, images_all, preds_all
    )
    imedian = n_outputs // 2
    mask = labels_all.squeeze() > 0
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
    ncfile: Path, labels: np.ndarray, images: np.ndarray, preds: np.ndarray
):
    """write output to netcdf file"""

    nimages = labels.shape[0]
    x = labels.shape[1]
    y = labels.shape[2]
    ds = xr.Dataset(
        {
            "labels": (["nimages", "x", "y"], labels.squeeze()),
            "inputs": (["nimages", "x", "y", "ninputs"], images),
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


def plot_preds(
    predictor: UnetPredictor,
    test_data: tf.data.Dataset,
) -> None:
    """Plot distributions and scatter"""
    for input_data, label_data in test_data:
        pred = predictor.model(input_data).numpy()
        fig, ax = plt.subplots(1, 2)
        ax = ax.ravel()
        label_data = label_data.numpy().squeeze()
        label_mask = label_data > 0
        iq = len(settings.QUANTILES) // 2
        ax[0].scatter(label_data[label_mask], pred[:, :, :, iq][label_mask])
        ax[1].hist(
            label_data[label_mask],
            bins=100,
            histtype="step",
            density=True,
            label="labels",
        )
        ax[1].hist(
            pred[:, :, :, iq][label_mask],
            bins=100,
            histtype="step",
            density=True,
            label="predictions",
        )
        ax[1].legend()
        plt.show()
