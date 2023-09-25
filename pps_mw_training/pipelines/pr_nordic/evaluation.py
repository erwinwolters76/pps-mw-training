import matplotlib.pyplot as plt  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow import keras

from pps_mw_training.pipelines.pr_nordic import settings
from pps_mw_training.models.predictors.unet_predictor import UnetPredictor


VMIN = -30
VMAX = 60
CMAP = "coolwarm"
QUANTILE_IDXS = [3, 4, 5]


def evaluate_model(
    predictor: UnetPredictor,
    test_data: tf.data.Dataset,
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
