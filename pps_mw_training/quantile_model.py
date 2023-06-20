from dataclasses import dataclass
from pathlib import Path
from typing import cast, Any, List, Dict
import json

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from xarray import Dataset  # type: ignore

from pps_mw_training.scaler import Scaler


@dataclass
class QuantileModel:
    """Quantile regression neural network model object."""

    model: keras.Sequential
    pre_scaler: Scaler
    post_scaler: Scaler
    input_params: List[str]
    output_params: List[str]
    quantiles: List[float]

    @classmethod
    def load(
        cls,
        model_config_file: Path,
    ) -> "QuantileModel":
        """Load model from precalculated weights."""
        with open(model_config_file) as config_file:
            config = json.load(config_file)
        input_params = config["input_parameters"]
        output_params = config["output_parameters"]
        model = cls.create(
            len(input_params),
            len(output_params),
            config["n_hidden_layers"],
            config["n_neurons_per_layer"],
            config["activation"],
            config["quantiles"],
        )
        model.compile()
        model.load_weights(config["model_weights"])
        return cls(
            model,
            pre_scaler=Scaler.from_dict(input_params),
            post_scaler=Scaler.from_dict(output_params),
            input_params=[p["name"] for p in input_params],
            output_params=[p["name"] for p in output_params],
            quantiles=config["quantiles"],
        )

    @staticmethod
    def create(
        n_input_params: int,
        n_output_params: int,
        n_hidden_layers: int,
        n_neurons_per_layer: int,
        activation: str,
        quantiles: List[float],
    ) -> keras.Sequential:
        """Create the quantile model."""
        model = keras.Sequential()
        model.add(keras.Input(shape=(n_input_params,)))
        for _ in range(n_hidden_layers):
            model.add(
                keras.layers.Dense(
                    n_neurons_per_layer, activation=activation,
                )
            )
        model.add(
            keras.layers.Dense(
                n_output_params * len(quantiles), activation="linear"
            )
        )
        model.summary()
        return model

    @classmethod
    def train(
        cls,
        input_parameters: List[Dict[str, Any]],
        output_parameters: List[Dict[str, Any]],
        n_hidden_layers: int,
        n_neurons_per_layer: int,
        activation: str,
        quantiles: List[float],
        training_data: Dataset,
        validation_data: Dataset,
        batch_size: int,
        epochs: int,
        initial_learning_rate: float,
        first_decay_steps: int,
        t_mul: float,
        m_mul: float,
        alpha: float,
        output_path: Path,
    ) -> None:
        """Run the training pipeline fro the quantile model."""
        input_params = [cast(str, p["name"]) for p in input_parameters]
        output_params = [cast(str, p["name"]) for p in output_parameters]
        n_inputs = len(input_params)
        n_outputs = len(output_params)
        model = cls.create(
            n_inputs,
            n_outputs,
            n_hidden_layers,
            n_neurons_per_layer,
            activation,
            quantiles,
        )
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
                n_outputs, quantiles, y_true, y_pred
            ),
            metrics=['accuracy'],
        )
        input_scaler = Scaler.from_dict(input_parameters)
        output_scaler = Scaler.from_dict(output_parameters)
        output_path.mkdir(parents=True, exist_ok=True)
        weights_file = output_path / "weights.h5"
        model.fit(
            tf.data.Dataset.from_tensor_slices(
                (
                    cls.prescale(training_data, input_scaler, input_params),
                    cls.prescale(training_data, output_scaler, output_params),
                )
            ).batch(batch_size=batch_size),
            epochs=epochs,
            verbose=1,
            validation_data=tf.data.Dataset.from_tensor_slices(
                (
                    cls.prescale(validation_data, input_scaler, input_params),
                    cls.prescale(validation_data, output_scaler, output_params),
                )
            ).batch(batch_size=batch_size),
            callbacks=[
                ModelCheckpoint(
                    weights_file,
                    save_best_only=True,
                    save_weights_only=True,
                )
            ],
        )
        with open(output_path / "network_config.json", "w") as outfile:
            outfile.write(
                json.dumps(
                    {
                        "input_parameters": input_parameters,
                        "output_parameters": output_parameters,
                        "n_hidden_layers": n_hidden_layers,
                        "n_neurons_per_layer": n_neurons_per_layer,
                        "activation": activation,
                        "quantiles": quantiles,
                        "model_weights": weights_file.as_posix(),
                    },
                    indent=4,
                )
            )

    @staticmethod
    def prescale(
        data: Dataset,
        pre_scaler: Scaler,
        input_params: List[str],
    ) -> np.ndarray:
        """Prescale data."""
        return pre_scaler.apply(
            np.column_stack([data[param].values for param in input_params])
        )

    def postscale(
        self,
        data: np.ndarray,
    ) -> Dataset:
        """Transform numpy array holding retrieval data to a dataset."""
        n = len(self.quantiles)
        return Dataset(
            data_vars={
                param: (
                    ("t", "quantile"),
                    self.post_scaler.reverse(
                        data[:, idx * n:  (idx + 1) * n],
                        idx=idx,
                    )
                ) for idx, param in enumerate(self.output_params)
            },
            coords={
                "quantile": ("quantile", self.quantiles)
            },
        )

    def predict(
        self,
        input_data: Dataset,
    ) -> Dataset:
        """Apply the trained neural network for a retrieval purpose."""
        prescaled = self.prescale(
            input_data, self.pre_scaler, self.input_params,
        )
        predicted = self.model(prescaled)
        return self.postscale(predicted.numpy())


def quantile_loss(
    n_params: int,
    quantiles: List[float],
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:
    """Quantile loss function handling multiple quantiles and parameters."""
    s = len(quantiles)
    q = tf.constant(np.tile(quantiles, n_params), dtype=tf.float32)
    e = tf.concat(
        [
            tf.expand_dims(y_true[:, i], 1) - y_pred[:, i * s: (i + 1) * s]
            for i in range(n_params)
        ],
        axis=1
    )
    v = tf.maximum(q * e, (q - 1) * e)
    return tf.reduce_mean(v)
