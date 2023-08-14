from pathlib import Path
import os

from pps_mw_training.data.nordic_precip import (
    get_training_dataset, prepare_dataset
)
from pps_mw_training.evaluation.unet_model_evaluation import evaluate_model
from pps_mw_training.unet_model import UNetModel


MODEL_WEIGHTS = Path(
    os.environ.get("MODEL_WEIGHTS", "/tmp/pretrained_weights.h5")
)
N_INPUTS = 5
QUANTILES = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
N_OUTPUTS = len(QUANTILES)
N_UNET_BASE = 16
N_FEATURES = 128
N_LAYERS = 4
N_EPOCHS = 10
BATCH_SIZE = 64


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_training_dataset()
    train_ds = prepare_dataset(train_ds, N_INPUTS, BATCH_SIZE, augment=True)
    val_ds = prepare_dataset(val_ds, N_INPUTS, BATCH_SIZE, augment=True)
    test_ds = prepare_dataset(test_ds, N_INPUTS, BATCH_SIZE, augment=False)
    UNetModel.train(
        N_INPUTS,
        N_UNET_BASE,
        N_FEATURES,
        N_LAYERS,
        QUANTILES,
        train_ds,
        val_ds,
        N_EPOCHS,
        MODEL_WEIGHTS,
    )
    unet_model = UNetModel.load(
        N_INPUTS,
        N_OUTPUTS,
        N_UNET_BASE,
        N_FEATURES,
        N_LAYERS,
        MODEL_WEIGHTS,
    )
    evaluate_model(unet_model, test_ds)
