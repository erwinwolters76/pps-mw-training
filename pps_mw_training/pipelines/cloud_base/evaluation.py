from pathlib import Path
import json

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow import keras
import pickle
from pps_mw_training.pipelines.cloud_base import settings
from pps_mw_training.models.predictors.unet_predictor import UnetPredictor


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
    stats = get_stats(predictor, test_data)
    with open(output_path / "retrieval_stats.json", "w") as outfile:
        outfile.write(json.dumps(stats, indent=4))
    plot_preds(predictor, test_data)


def get_stats(
    predictor: UnetPredictor,
    test_data: tf.data.Dataset,
) -> dict[str, float]:
    """Get stats."""
    preds = []
    labels = []
    images = []

    for input_data, label_data in test_data:
        print(input_data.numpy().shape)
        for p in predictor.model(input_data).numpy():
            preds.append(p)
        for r in label_data.numpy():
            labels.append(r)
        for i in input_data.numpy():
            images.append(i)

    preds_all = np.expand_dims(
        np.array(preds)[:, :, :, len(settings.QUANTILES) // 2], axis=3
    )
    labels_all = np.array(labels)
    images_all = np.array(images)

    with open("predictions.pickle", "wb") as f:
        pickle.dump(labels_all, f)
        pickle.dump(images_all, f)
        pickle.dump(preds_all, f)

    mask = labels_all > 0
    return {
        "rmse": float(
            np.sqrt(np.nanmean((preds_all[mask] - labels_all[mask]) ** 2))
        ),
        "mae": float(np.mean(np.abs(preds_all[mask] - labels_all[mask]))),
        "corr": float(np.corrcoef(preds_all[mask], labels_all[mask])[0, 1]),
    }


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
        fig.savefig("test.png")
