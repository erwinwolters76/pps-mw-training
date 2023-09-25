from pathlib import Path
import json

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow import keras

from pps_mw_training.pipelines.pr_nordic import settings
from pps_mw_training.models.predictors.unet_predictor import UnetPredictor


VMIN = -30
VMAX = 60
CMAP = "coolwarm"
QUANTILE_IDXS = [3, 4, 5]
DBZ_MIN = 15


def evaluate_model(
    predictor: UnetPredictor,
    test_data: tf.data.Dataset,
    output_path: Path,
) -> None:
    """Evaluata model, very rudimentary for the moment."""
    predictor.model.summary()
    keras.utils.plot_model(
        predictor.model.build_graph(128, len(settings.INPUT_PARAMS)),
        to_file=settings.MODEL_CONFIG_PATH / "model.png",
        show_shapes=True,
        show_dtype=False,
        show_layer_names=False,
        rankdir='TB',
        expand_nested=True,
        dpi=96,
        layer_range=None,
        show_layer_activations=True,
    )
    stats = get_stats(predictor, test_data)
    with open(output_path / "retrieval_stats.json", "w") as outfile:
        outfile.write(json.dumps(stats, indent=4))
    plot_stats(predictor, test_data)


def get_stats(
    predictor: UnetPredictor,
    test_data: tf.data.Dataset,
) -> dict[str, float]:
    """Get stats."""
    preds = []
    labels = []
    images = []
    for mw_data, radar_data in test_data:
        for p in predictor.model(mw_data).numpy():
            preds.append(p)
        for r in radar_data.numpy():
            labels.append(r)
        for i in mw_data.numpy():
            i_full = np.full(
                radar_data.numpy()[0].shape,
                settings.FILL_VALUE_IMAGES
            )
            i_full[0::2, 0::2, 0] = i[:, :, 0]
            images.append(i_full)

    preds = np.array(preds)[:, :, :, len(settings.QUANTILES) // 2]
    labels = np.array(labels)[:, :, :, 0]
    images = np.array(images)[:, :, :, 0]
    filt = (
        (labels != settings.FILL_VALUE_LABELS)
        & (images != settings.FILL_VALUE_IMAGES)
    )
    filt_dbz = labels[filt] >= DBZ_MIN
    return {
        "rmse": float(
            np.sqrt(np.mean((preds[filt] - labels[filt]) ** 2))
        ),
        "corr": float(np.corrcoef(preds[filt], labels[filt])[0, 1]),
        "pod": float(
            np.count_nonzero(preds[filt][filt_dbz] >= DBZ_MIN)
            / np.count_nonzero(filt_dbz)
        ),
        "far": float(
            np.count_nonzero(preds[filt][~filt_dbz] >= DBZ_MIN)
            / np.count_nonzero(~filt_dbz)
        ),
    }


def plot_stats(
    predictor: UnetPredictor,
    test_data: tf.data.Dataset,
) -> None:
    """Plot stats."""
    for mw_data, radar_data in test_data:
        pred = predictor.model(mw_data).numpy()
        for idx in range(pred.shape[0]):
            plt.figure()
            plt.subplot(2, 3, 1)
            plt.imshow(mw_data[idx, :, :, 0], cmap=CMAP)
            plt.colorbar()
            plt.title("MW_160")
            plt.subplot(2, 3, 2)
            plt.imshow(mw_data[idx, :, :, -1], cmap=CMAP)
            plt.colorbar()
            plt.title("MW_183")
            plt.subplot(2, 3, 3)
            plt.imshow(radar_data[idx], cmap=CMAP, vmin=VMIN, vmax=VMAX)
            plt.colorbar()
            plt.title("Radar")
            for i, qidx in enumerate(QUANTILE_IDXS):
                plt.subplot(2, 3, 4 + i)
                plt.imshow(
                    pred[idx, :, :, qidx],
                    cmap=CMAP,
                    vmin=VMIN,
                    vmax=VMAX,
                )
                plt.title(f"Quantile: {settings.QUANTILES[qidx]}")
                plt.colorbar()
            plt.show()
