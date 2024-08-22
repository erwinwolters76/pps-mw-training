from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np  # type: ignore
from xarray import Dataset  # type: ignore

from pps_mw_training.models.unet_model import UnetModel
from pps_mw_training.utils.scaler import Scaler


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
            # config["super_resolution"]
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
                    data[p["name"]][:, :, :].values, idx)
                for idx, p in enumerate(input_params)
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
        print("applying scaling")
        prescaled = self.prescale(
            input_data, self.pre_scaler, self.input_params, self.fill_value
        )
        return self.model(prescaled).numpy()
