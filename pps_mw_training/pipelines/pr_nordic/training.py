from pathlib import Path

from pps_mw_training.models.unet_model import UNetModel
from pps_mw_training.pipelines.pr_nordic import evaluation
from pps_mw_training.pipelines.pr_nordic import settings
from pps_mw_training.pipelines.pr_nordic import training_data
from pps_mw_training.utils.data import prepare_dataset


def train(
    n_layers: int,
    n_features: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    batch_size: int,
    n_epochs: int,
    model_weights: Path,
    only_evaluate: bool,
):
    "Run the Nordic precip training pipeline.",
    train_ds, val_ds, test_ds = training_data.get_training_dataset(
        train_fraction,
        validation_fraction,
        test_fraction,
    )
    if not only_evaluate:
        UNetModel.train(
            settings.N_INPUTS,
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
            model_weights,
        )
    unet_model = UNetModel.load(
        settings.N_INPUTS,
        settings.N_OUTPUTS,
        settings.N_UNET_BASE,
        n_features,
        n_layers,
        model_weights,
    )
    test_data = prepare_dataset(
        test_ds,
        settings.N_INPUTS,
        batch_size,
        augment=False,
    )
    evaluation.evaluate_model(unet_model, test_data)
