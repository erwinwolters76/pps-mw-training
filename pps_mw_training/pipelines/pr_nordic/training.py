from pathlib import Path

from pps_mw_training.models.trainers.unet_trainer import UnetTrainer
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
    "Run the Nordic precip training pipeline."
    train_ds, val_ds, test_ds = training_data.get_training_dataset(
        training_data_path,
        train_fraction,
        validation_fraction,
        test_fraction,
        settings.BATCH_SIZE,
        settings.MIN_QUALITY,
        settings.MAX_DISTANCE,
        settings.INPUT_PARAMS,
        settings.FILL_VALUE_IMAGES,
        settings.FILL_VALUE_LABELS,
    )
    if not only_evaluate:
        UnetTrainer.train(
            settings.INPUT_PARAMS,
            settings.N_UNET_BASE,
            settings.N_UNET_BLOCKS,
            n_features,
            n_layers,
            settings.QUANTILES,
            train_ds,
            val_ds,
            n_epochs,
            settings.FILL_VALUE_IMAGES,
            settings.FILL_VALUE_LABELS,
            settings.IMAGE_SIZE,
            settings.INITIAL_LEARNING_RATE,
            settings.FIRST_DECAY_STEPS,
            settings.T_MUL,
            settings.M_MUL,
            settings.ALPHA,
            model_config_path,
        )
    model = UnetTrainer.load(model_config_path / "network_config.json")
    evaluation.evaluate_model(model, test_ds)
