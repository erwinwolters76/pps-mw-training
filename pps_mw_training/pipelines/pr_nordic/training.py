from pathlib import Path

from pps_mw_training.models.unet_model import UNetModel
from pps_mw_training.pipelines.pr_nordic import evaluation
from pps_mw_training.pipelines.pr_nordic import settings
from pps_mw_training.pipelines.pr_nordic import training_data


def train(
    n_layers: int,
    n_features: int,
    training_data_path: Path,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    batch_size: int,
    n_epochs: int,
    model_config_path: Path,
    only_evaluate: bool,
):
    "Run the Nordic precip training pipeline.",
    train_ds, val_ds, test_ds = training_data.get_training_dataset(
        training_data_path,
        train_fraction,
        validation_fraction,
        test_fraction,
        settings.CHANNELS,
    )
    if not only_evaluate:
        UNetModel.train(
            len(settings.CHANNELS),
            settings.N_UNET_BASE,
            n_features,
            n_layers,
            settings.QUANTILES,
            train_ds,
            val_ds,
            batch_size,
            n_epochs,
            settings.INITIAL_LEARNING_RATE,
            settings.FIRST_DECAY_STEPS,
            settings.T_MUL,
            settings.M_MUL,
            settings.ALPHA,
            model_config_path,
        )
    unet_model = UNetModel.load(model_config_path / "network_config.json")
    evaluation.evaluate_model(unet_model, test_ds)
