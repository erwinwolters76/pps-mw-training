import matplotlib.pyplot as plt  # type: ignore
from tensorflow import keras  # type: ignore
from xarray import Dataset, DataArray  # type: ignore

from pps_mw_training.pipelines.pr_nordic import settings
from pps_mw_training.models.predictors.unet_predictor import UnetPredictor


VMIN = -20
VMAX = 10
CMAP = "coolwarm"
QUANTILE_IDXS = [3, 4, 5]


def evaluate_model(
    predictor: UnetPredictor,
    mw_data: Dataset,
    radar_data: DataArray,
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
    pred = predictor.predict(mw_data)
    for idx in range(mw_data.time.size):
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.imshow(mw_data["mw_160"][idx, :, :, 0], cmap=CMAP)
        plt.colorbar()
        plt.title("MW_160")
        plt.subplot(2, 3, 2)
        plt.imshow(mw_data["mw_183"][idx, :, :, 2], cmap=CMAP)
        plt.colorbar()
        plt.title("MW_183")
        plt.subplot(2, 3, 3)
        plt.imshow(radar_data.data[idx], cmap=CMAP, vmin=VMIN, vmax=VMAX)
        plt.colorbar()
        plt.title("Radar")
        for i, qidx in enumerate(QUANTILE_IDXS):
            plt.subplot(2, 3, 4 + i)
            plt.imshow(pred[idx, :, :, qidx], cmap=CMAP, vmin=VMIN, vmax=VMAX)
            plt.title(f"Quantile: {settings.QUANTILES[qidx]}")
            plt.colorbar()
        plt.show()
