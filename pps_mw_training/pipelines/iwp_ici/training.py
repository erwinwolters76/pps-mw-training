from pathlib import Path

from pps_mw_training.models.quantile_model import QuantileModel
from pps_mw_training.pipelines.iwp_ici import evaluation
from pps_mw_training.pipelines.iwp_ici import settings
from pps_mw_training.pipelines.iwp_ici import training_data


def train(
    n_hidden_layers: int,
    n_neurons_per_hidden_layer: int,
    activation: str,
    ici_db_file: Path,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    batch_size: int,
    n_epochs: int,
    missing_fraction: float,
    model_config_path: Path,
    only_evaluate: bool,
) -> None:
    """Run the IWP ICI training pipeline"""
    train_data, test_data, val_data = training_data.get_training_data(
        ici_db_file,
        train_fraction,
        validation_fraction,
        test_fraction,
        settings.INPUT_PARAMS,
        settings.NOISE,
    )
    if not only_evaluate:
        QuantileModel.train(
            settings.INPUT_PARAMS,
            settings.OUTPUT_PARAMS,
            n_hidden_layers,
            n_neurons_per_hidden_layer,
            activation,
            settings.QUANTILES,
            train_data,
            val_data,
            batch_size,
            n_epochs,
            settings.INITIAL_LEARNING_RATE,
            settings.FIRST_DECAY_STEPS,
            settings.T_MUL,
            settings.M_MUL,
            settings.ALPHA,
            missing_fraction,
            settings.FILL_VALUE,
            model_config_path,
        )
    model = QuantileModel.load(model_config_path / "network_config.json")
    evaluation.evaluate_model(
        model, test_data, missing_fraction, model_config_path
    )
