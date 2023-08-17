from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow import keras
from xarray import Dataset, DataArray  # type: ignore

from pps_mw_training.utils.blocks import (
    ConvolutionBlock,
    DownsamplingBlock,
    MLP,
    UpsamplingBlock,
)
from pps_mw_training.utils.data import random_crop_and_flip
from pps_mw_training.utils.loss_function import quantile_loss
from pps_mw_training.utils.scaler import Scaler


class UNetBaseModel(keras.Model):
    """U-Net convolutinal neural network object."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_unet_base: int,
        n_features: int,
        n_layers: int,
    ):
        super().__init__()
        n = n_unet_base
        self.in_block = ConvolutionBlock(n_inputs, n)
        self.down_block_1 = DownsamplingBlock(n * 1, n * 2)
        self.down_block_2 = DownsamplingBlock(n * 2, n * 4)
        self.down_block_3 = DownsamplingBlock(n * 4, n * 8)
        self.down_block_4 = DownsamplingBlock(n * 8, n * 16)
        self.up_block_1 = UpsamplingBlock(n * 16, n * 8)
        self.up_block_2 = UpsamplingBlock(n * 8, n * 4)
        self.up_block_3 = UpsamplingBlock(n * 4, n * 2)
        self.up_block_4 = UpsamplingBlock(n * 2, n * 1)
        self.out_block = MLP(n_outputs, n_features, n_layers)

    def call(self, inputs):
        d_0 = self.in_block(inputs)
        d_1 = self.down_block_1(d_0)
        d_2 = self.down_block_2(d_1)
        d_3 = self.down_block_3(d_2)
        d_4 = self.down_block_4(d_3)
        u = self.up_block_1([d_4, d_3])
        u = self.up_block_2([u, d_2])
        u = self.up_block_3([u, d_1])
        u = self.up_block_4([u, d_0])
        return self.out_block(u)

    def build_graph(self, image_size: int, n_inputs: int):
        x = keras.Input(shape=(image_size, image_size, n_inputs))
        return keras.Model(inputs=[x], outputs=self.call(x))


@dataclass
class UNetModel:
    """
    Object for handling training, loading, and predictions of a
    quantile regression U-Net convolutional neural network model.
    """
    model: UNetBaseModel
    pre_scaler: Scaler
    input_params: list[dict[str, Any]]
    fill_value: float

    @classmethod
    def load(
        cls,
        model_config_file: Path,
    ) -> "UNetModel":
        """Load the model from config file."""
        with open(model_config_file) as config_file:
            config = json.load(config_file)
        input_parameters = config["input_parameters"]
        n_inputs = len(input_parameters)
        n_outputs = len(config["quantiles"])
        model = UNetBaseModel(
            n_inputs,
            n_outputs,
            config["n_unet_base"],
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

    @classmethod
    def train(
        cls,
        input_parameters: list[dict[str, Any]],
        n_unet_base: int,
        n_features: int,
        n_layers: int,
        quantiles: list[float],
        training_data: tuple[Dataset, DataArray],
        validation_data: tuple[Dataset, DataArray],
        batch_size: int,
        n_epochs: int,
        fill_value: float,
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
            model = UNetBaseModel(
                n_inputs, n_outputs, n_unet_base, n_features, n_layers,
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
                1, quantiles, y_true, y_pred,
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
                fill_value,
            ),
            epochs=n_epochs,
            validation_data=cls.prepare_dataset(
                input_parameters,
                validation_data,
                batch_size,
                image_size,
                fill_value,
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
                        "n_features": n_features,
                        "n_layers": n_layers,
                        "quantiles": quantiles,
                        "fill_value": fill_value,
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
        fill_value: float,
    ) -> tf.data.Dataset:
        """Prepare dataset for training."""
        input_scaler = Scaler.from_dict(input_parameters)
        images = cls.prescale(
            training_data[0],
            input_scaler,
            input_parameters,
            fill_value,
        )
        labels = np.expand_dims(training_data[1].values, axis=3)
        labels[~np.isfinite(labels)] = fill_value
        return random_crop_and_flip(images, labels, image_size, batch_size)

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
