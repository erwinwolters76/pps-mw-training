import matplotlib.pyplot as plt  # type: ignore

from tensorflow import keras  # type: ignore
from xarray import Dataset, DataArray  # type: ignore

from pps_mw_training.models.unet_model import UNetModel
from pps_mw_training.pipelines.pr_nordic import settings


def evaluate_model(
    unet_model: UNetModel,
    mw_data: Dataset,
    radar_data: DataArray,
) -> None:
    """Evaluata model, very rudimentary for the moment."""
    unet_model.model.summary()
    keras.utils.plot_model(
        unet_model.model.build_graph(128, len(settings.INPUT_PARAMS)),
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
    pred = unet_model.predict(mw_data)
    for idx in range(mw_data.time.size):
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(mw_data["mw_183"][idx, :, :, 3], cmap="coolwarm")
        plt.colorbar()
        plt.title("MW")
        plt.subplot(2, 2, 2)
        plt.imshow(radar_data.data[idx], cmap="coolwarm", vmin=-30, vmax=30)
        plt.colorbar()
        plt.title("Radar")
        plt.subplot(2, 3, 4)
        plt.imshow(pred[idx, :, :, 3], cmap="coolwarm")
        plt.title("0.25")
        plt.colorbar()
        plt.subplot(2, 3, 5)
        plt.imshow(pred[idx, :, :, 4], cmap="coolwarm")
        plt.colorbar()
        plt.title("0.5")
        plt.subplot(2, 3, 6)
        plt.imshow(pred[idx, :, :, 5], cmap="coolwarm")
        plt.colorbar()
        plt.title("0.75")
        plt.show()
