from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import tensorflow as tf  # type: ignore
from tensorflow import keras

from pps_mw_training.models.unet_model import UnetModel
from pps_mw_training.models.predictors.unet_predictor import UnetPredictor
from pps_mw_training.models.trainers.utils import MemoryUsageCallback
from pps_mw_training.utils.augmentation import random_crop_and_flip
from pps_mw_training.utils.loss_function import quantile_loss
from pps_mw_training.utils.scaler import Scaler


@dataclass
class UnetTrainer(UnetPredictor):
    """
    Object for handling training of a quantile regression U-Net
    convolutional neural network model.
    """
    model: UnetModel
    pre_scaler: Scaler
    input_params: list[dict[str, Any]]
    fill_value: float

    @classmethod
    def train(
        cls,
        input_parameters: list[dict[str, Any]],
        n_unet_base: int,
        n_unet_blocks: int,
        n_features: int,
        n_layers: int,
        quantiles: list[float],
        training_data: tf.data.Dataset,
        validation_data: tf.data.Dataset,
        n_epochs: int,
        fill_value_images: float,
        fill_value_labels: float,
        image_size: int,
        initial_learning_rate: float,
        decay_steps_factor: float,
        alpha: float,
        output_path: Path,
    ) -> None:
        """Train the model."""
        model_config_file = output_path / "network_config.json"
        if model_config_file.is_file():
            # load and continue the training of an existing model
            model = cls.load(model_config_file).model
        else:
            n_inputs = len(input_parameters)
            n_outputs = len(quantiles)
            model = UnetModel(
                n_inputs,
                n_outputs,
                n_unet_base,
                n_unet_blocks,
                n_features,
                n_layers,
            )
            model.build_graph(image_size, n_inputs)
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=int(
                decay_steps_factor * len(training_data) * n_epochs
            ),
            alpha=alpha,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=lambda y_true, y_pred: quantile_loss(
                1, quantiles, y_true, y_pred, fill_value=fill_value_labels,
            ),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        weights_file = output_path / "pr_nordic.weights.h5"
        training_data = training_data.map(
            lambda x, y: random_crop_and_flip(x, y, tf.constant(image_size))
        )
        validation_data = validation_data.map(
            lambda x, y: random_crop_and_flip(x, y, tf.constant(image_size))
        )
        validation_data = validation_data.cache()
        history = model.fit(
            training_data,
            epochs=n_epochs,
            validation_data=validation_data,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    weights_file,
                    save_best_only=True,
                    save_weights_only=True,
                ),
                MemoryUsageCallback(),
            ],
        )
        with open(output_path / "fit_history.json", "w") as outfile:
            outfile.write(json.dumps(history.history, indent=4))
        with open(model_config_file, "w") as outfile:
            outfile.write(
                json.dumps(
                    {
                        "input_parameters": input_parameters,
                        "n_unet_base": n_unet_base,
                        "n_unet_blocks": n_unet_blocks,
                        "n_features": n_features,
                        "n_layers": n_layers,
                        "quantiles": quantiles,
                        "fill_value": fill_value_images,
                        "model_weights": weights_file.as_posix(),
                    },
                    indent=4,
                )
            )
