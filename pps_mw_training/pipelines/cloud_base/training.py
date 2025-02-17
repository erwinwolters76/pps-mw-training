from pathlib import Path
from typing import Optional
from pps_mw_training.models.trainers.unet_trainer import UnetTrainer
from pps_mw_training.pipelines.cloud_base import evaluation
from pps_mw_training.pipelines.cloud_base import settings
from pps_mw_training.pipelines.cloud_base import training_data


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
    file_limit: Optional[int],
):
    "Run the cloud base training pipeline."
    train_ds, val_ds, test_ds = training_data.get_training_dataset(
        training_data_path,
        train_fraction,
        validation_fraction,
        test_fraction,
        batch_size,
        settings.INPUT_PARAMS,
        settings.LABEL_PARAMS,
        settings.FILL_VALUE_IMAGES,
        settings.FILL_VALUE_LABELS,
        file_limit,
    )
    if not only_evaluate:
        UnetTrainer.train(
            settings.INPUT_PARAMS,
            settings.N_UNET_BASE,
            settings.N_UNET_BLOCKS,
            n_features,
            n_layers,
            settings.SUPER_RESOLUTION,
            settings.QUANTILES,
            train_ds,
            val_ds,
            n_epochs,
            settings.FILL_VALUE_IMAGES,
            settings.FILL_VALUE_LABELS,
            settings.IMAGE_SIZE,
            settings.AUGMENTATION_TYPE,
            settings.INITIAL_LEARNING_RATE,
            settings.DECAY_STEPS_FACTOR,
            settings.ALPHA,
            model_config_path,
        )
    model = UnetTrainer.load(model_config_path / "network_config.json")
    evaluation.evaluate_model(model, test_ds, model_config_path)
