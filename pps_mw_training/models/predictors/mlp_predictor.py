from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np  # type: ignore
from xarray import Dataset  # type: ignore

from pps_mw_training.models.mlp_model import MlpModel
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
    input_params: list[str]
    output_params: list[str]
    quantiles: list[float]
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
        input_params: list[str],
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
