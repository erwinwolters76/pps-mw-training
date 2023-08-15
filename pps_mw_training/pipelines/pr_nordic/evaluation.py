import matplotlib.pyplot as plt  # type: ignore

import tensorflow as tf  # type: ignore
from tensorflow import keras

from pps_mw_training.models.unet_model import UNetModel
from pps_mw_training.pipelines.pr_nordic import settings
from pps_mw_training.utils.data import prepare_dataset


def evaluate_model(
    unet_model: UNetModel,
    dataset: tf.data.Dataset,
) -> None:
    """Evaluata model, very rudimentary for the moment."""
    unet_model.model.summary()
    keras.utils.plot_model(
        unet_model.model.build_graph(128, settings.N_INPUTS),
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
    test_data = prepare_dataset(
        dataset,
        settings.N_INPUTS,
        settings.BATCH_SIZE,
        augment=False,
    )
    for images, masks in test_data.take(1):
        preds = unet_model.predict(images)
        for pred, mask, image in zip(preds, masks, images):
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(image[:, :, -1], vmin=0, vmax=1, cmap="coolwarm")
            plt.title("MW")
            plt.subplot(2, 2, 2)
            plt.imshow(mask, vmin=-1, vmax=1, cmap="coolwarm")
            plt.title("Radar")
            plt.subplot(2, 3, 4)
            plt.imshow(pred[:, :, 3], cmap="coolwarm", vmin=-1, vmax=1)
            plt.title("0.25")
            plt.colorbar()
            plt.subplot(2, 3, 5)
            plt.imshow(pred[:, :, 4], cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar()
            plt.title("0.5")
            plt.subplot(2, 3, 6)
            plt.imshow(pred[:, :, 5], cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar()
            plt.title("0.75")
            plt.show()
