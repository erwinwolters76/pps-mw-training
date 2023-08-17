from dataclasses import dataclass
from pathlib import Path
from typing import cast, Any, List, Dict
import json

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from xarray import Dataset  # type: ignore

from pps_mw_training.models.mlp_model import MlpModel
from pps_mw_training.utils.augmentation import set_missing_data
from pps_mw_training.utils.loss_function import quantile_loss
from pps_mw_training.utils.scaler import Scaler


@dataclass
class MlpPredictor:
    """
    Object for handling the loading and processing of a quantile
    regression multi layer perceptron neural network model.
    """

    model: MlpModel
    pre_scaler: Scaler
    post_scaler: Scaler
    input_params: List[str]
    output_params: List[str]
    quantiles: List[float]
    fill_value: float

    @classmethod
    def load(
        cls,
        model_config_file: Path,
    ) -> "MlpPredictor":
        """Load model from config file."""
        with open(model_config_file) as config_file:
            config = json.load(config_file)
        input_params = config["input_parameters"]
        output_params = config["output_parameters"]
        quantiles = config["quantiles"]
        model = MlpModel(
            len(input_params),
            len(output_params) * len(quantiles),
            config["n_hidden_layers"],
            config["n_neurons_per_layer"],
            config["activation"],
        )
        model.summary()
        model.compile()
        model.load_weights(config["model_weights"])
        return cls(
            model,
            pre_scaler=Scaler.from_dict(input_params),
            post_scaler=Scaler.from_dict(output_params),
            input_params=[p["name"] for p in input_params],
            output_params=[p["name"] for p in output_params],
            quantiles=quantiles,
            fill_value=config["fill_value"]
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
        """Postscale data."""
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
        """Predict output fron input data."""
        prescaled = self.prescale(
            input_data, self.pre_scaler, self.input_params,
        )
        prescaled[~np.isfinite(prescaled)] = self.fill_value
        predicted = self.model(prescaled)
        return self.postscale(predicted.numpy())


@dataclass
class MlpTrainer(MlpPredictor):
    """
    Object for handling training of a quantile regression multi layer
    perceptron neural network model.
    """

    model: MlpModel
    pre_scaler: Scaler
    post_scaler: Scaler
    input_params: List[str]
    output_params: List[str]
    quantiles: List[float]
    fill_value: float

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
        missing_fraction: float,
        fill_value: float,
        output_path: Path,
    ) -> None:
        """Run the training pipeline for the model."""
        model = MlpModel(
            len(input_parameters),
            len(output_parameters) * len(quantiles),
            n_hidden_layers,
            n_neurons_per_layer,
            activation,
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
                len(output_parameters), quantiles, y_true, y_pred
            ),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        weights_file = output_path / "weights.h5"
        history = model.fit(
            cls.prepare_data(
                input_parameters,
                output_parameters,
                training_data,
                batch_size,
                missing_fraction,
                fill_value,
            ),
            epochs=epochs,
            validation_data=cls.prepare_data(
                input_parameters,
                output_parameters,
                validation_data,
                batch_size,
                missing_fraction,
                fill_value,
            ),
            callbacks=[
                ModelCheckpoint(
                    weights_file,
                    save_best_only=True,
                    save_weights_only=True,
                )
            ],
        )
        with open(output_path / "fit_history.json", "w") as outfile:
            outfile.write(json.dumps(history.history, indent=4))
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
                        "fill_value": fill_value,
                        "model_weights": weights_file.as_posix(),
                    },
                    indent=4,
                )
            )

    @classmethod
    def prepare_data(
        cls,
        input_parameters: list[dict[str, Any]],
        output_parameters: List[Dict[str, Any]],
        training_data: Dataset,
        batch_size: int,
        missing_fraction: float,
        fill_value: float,
    ) -> tf.data.Dataset:
        """Prepare data for training."""
        input_scaler = Scaler.from_dict(input_parameters)
        output_scaler = Scaler.from_dict(output_parameters)
        input_params = [cast(str, p["name"]) for p in input_parameters]
        output_params = [cast(str, p["name"]) for p in output_parameters]
        return tf.data.Dataset.from_tensor_slices(
            (
                cls.prescale(training_data, input_scaler, input_params),
                cls.prescale(training_data, output_scaler, output_params),
            )
        ).batch(
            batch_size=batch_size
        ).map(
            lambda x, y: (set_missing_data(x, missing_fraction, fill_value), y)
        )
