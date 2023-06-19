from dataclasses import dataclass
from pathlib import Path
from typing import cast, Any, List, Dict, Tuple
import json

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from xarray import Dataset  # type: ignore

from pps_mw_training.scaler import Scaler
from pps_mw_training.utils import as_array


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
        quantiles = config["quantiles"]
        n_neurons = config["n_neurons_per_layer"]
        activation = config["activation"]
        n_layers = config["n_hidden_layers"]
        layers: List[Tuple[int, str]] = [
            (n_neurons, activation) for _ in range(n_layers)
        ] + [
            (len(output_params) * len(quantiles), "linear")
        ]
        model = cls.create(len(input_params), layers)
        model.compile()
        model.load_weights(config["model_weights"])
        return cls(
            model,
            pre_scaler=Scaler.from_dict(input_params),
            post_scaler=Scaler.from_dict(output_params),
            input_params=[p["name"] for p in input_params],
            output_params=[p["name"] for p in output_params],
            quantiles=quantiles,
        )

    @staticmethod
    def create(
        n_input_params: int,
        layers: List[Tuple[int, str]],
    ) -> keras.Sequential:
        """Create the model."""
        model = keras.Sequential()
        model.add(keras.Input(shape=(n_input_params,)))
        for n_neurons, activation in layers:
            model.add(keras.layers.Dense(n_neurons, activation=activation))
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
        layers: List[Tuple[int, str]] = [
            (n_neurons_per_layer, activation) for _ in range(n_hidden_layers)
        ] + [
            (n_outputs * len(quantiles), "linear")
        ]
        model = cls.create(n_inputs, layers)
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
        weights_file = output_path / "weights.h5"
        model.fit(
            tf.data.Dataset.from_tensor_slices(
                (
                    input_scaler.apply(
                        as_array(training_data, input_params)
                    ),
                    output_scaler.apply(
                        as_array(training_data, output_params)
                    )
                )
            ).batch(batch_size=batch_size),
            epochs=epochs,
            verbose=1,
            validation_data=tf.data.Dataset.from_tensor_slices(
                (
                    input_scaler.apply(
                        as_array(validation_data, input_params)
                    ),
                    output_scaler.apply(
                        as_array(validation_data, output_params)
                    ),
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

    def prescale(
        self,
        data: Dataset,
    ) -> np.ndarray:
        """Prescale data."""
        return self.pre_scaler.apply(as_array(data, self.input_params))

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
        prescaled = self.prescale(input_data)
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
