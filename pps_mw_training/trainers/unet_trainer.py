from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow import keras
from xarray import Dataset, DataArray  # type: ignore

from pps_mw_training.models.unet_model import UnetModel
from pps_mw_training.utils.augmentation import random_crop_and_flip
from pps_mw_training.utils.loss_function import quantile_loss
from pps_mw_training.utils.scaler import Scaler


AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class UnetPredictor:
    """
    Object for handling the loading and processing of a quantile
    regression U-Net convolutional neural network model.
    """
    model: UnetModel
    pre_scaler: Scaler
    input_params: list[dict[str, Any]]
    fill_value: float

    @classmethod
    def load(
        cls,
        model_config_file: Path,
    ) -> "UnetPredictor":
        """Load the model from config file."""
        with open(model_config_file) as config_file:
            config = json.load(config_file)
        input_parameters = config["input_parameters"]
        n_inputs = len(input_parameters)
        n_outputs = len(config["quantiles"])
        model = UnetModel(
            n_inputs,
            n_outputs,
            config["n_unet_base"],
            config["n_unet_blocks"],
            config["n_features"],
            config["n_layers"],
        )
        model.build((None, None, None, n_inputs))
        model.load_weights(config["model_weights"])
        return cls(
            model,
            Scaler.from_dict(input_parameters),
            input_parameters,
            config["fill_value"],
        )

    @staticmethod
    def prescale(
        data: Dataset,
        pre_scaler: Scaler,
        input_params: list[dict[str, Any]],
        fill_value: float,
    ) -> np.ndarray:
        """Prescale data."""
        data = np.stack(
            [
                pre_scaler.apply(
                    data[p["band"]][:, :, :, p["index"]].values,
                    idx,
                ) for idx, p in enumerate(input_params)
            ],
            axis=3,
        )
        data[~np.isfinite(data)] = fill_value
        return data

    def predict(
        self,
        input_data: Dataset,
    ) -> np.ndarray:
        """Apply the trained neural network for a retrieval purpose."""
        prescaled = self.prescale(
            input_data, self.pre_scaler, self.input_params, self.fill_value
        )
        return self.model(prescaled).numpy()


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
        training_data: tuple[Dataset, DataArray],
        validation_data: tuple[Dataset, DataArray],
        batch_size: int,
        n_epochs: int,
        fill_value_images: float,
        fill_value_labels: float,
        image_size: int,
        initial_learning_rate: float,
        first_decay_steps: int,
        t_mul: float,
        m_mul: float,
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
            model.build((None, None, None, n_inputs))
        learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=lambda y_true, y_pred: quantile_loss(
                1, quantiles, y_true, y_pred, fill_value=fill_value_labels,
            ),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        weights_file = output_path / "weights.h5"
        history = model.fit(
            cls.prepare_dataset(
                input_parameters,
                training_data,
                batch_size,
                image_size,
                fill_value_images,
                fill_value_labels,
            ),
            epochs=n_epochs,
            validation_data=cls.prepare_dataset(
                input_parameters,
                validation_data,
                batch_size,
                image_size,
                fill_value_images,
                fill_value_labels,
            ),
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    weights_file,
                    save_best_only=True,
                    save_weights_only=True,
                )
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

    @classmethod
    def prepare_dataset(
        cls,
        input_parameters: list[dict[str, Any]],
        training_data: tuple[Dataset, DataArray],
        batch_size: int,
        image_size: int,
        fill_value_images: float,
        fill_value_labels: float,
    ) -> tf.data.Dataset:
        """Prepare dataset for training."""
        input_scaler = Scaler.from_dict(input_parameters)
        images = cls.prescale(
            training_data[0],
            input_scaler,
            input_parameters,
            fill_value_images,
        )
        labels = np.expand_dims(training_data[1].values, axis=3)
        labels[~np.isfinite(labels)] = fill_value_labels
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        ds = ds.batch(batch_size)
        ds = ds.map(lambda x, y: random_crop_and_flip(x, y, image_size))
        ds.prefetch(buffer_size=AUTOTUNE)
        return ds
