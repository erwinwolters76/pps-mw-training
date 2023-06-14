#!/usr/bin/env python
from pathlib import Path

from tensorflow.keras.callbacks import ModelCheckpoint

from pps_mw_training import ici
from pps_mw_training.evaluation import evaluate_model
from pps_mw_training.model import create_model
from pps_mw_training.utils import split_dataset


# model parameters
QUANTILES = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
N_HIDDEN_LAYERS = 4
N_NEURONS = 128
ACTIVATION = "relu"
# training parameters
BATCH_SIZE = 4096
EPOCHS = 20
TRAIN_FRACTION = 0.7
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
MODEL_WEIGHTS = Path("saved_model/pretrained_weights.h5")


if __name__ == "__main__":
    # Create, train, and evaluate model
    TRAIN = True
    N_PARAMS = len(ici.STATE_PARAMS)
    N_CHANNELS = len(ici.CHANNEL_PARAMS)
    full_dataset = ici.load_retrieval_database()
    training_dataset, test_dataset, validation_dataset = split_dataset(
        full_dataset,
        TRAIN_FRACTION,
        VALIDATION_FRACTION,
        TEST_FRACTION,
    )
    model = create_model(
        N_CHANNELS,
        N_HIDDEN_LAYERS,
        N_NEURONS,
        ACTIVATION,
        N_PARAMS,
        QUANTILES,
    )
    if TRAIN:
        model.fit(
            training_dataset.batch(batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            verbose=1,
            validation_data=validation_dataset.batch(batch_size=BATCH_SIZE),
            callbacks=[
                ModelCheckpoint(
                    MODEL_WEIGHTS,
                    save_best_only=True,
                    save_weights_only=True,
                )
            ],
        )
    model.load_weights(MODEL_WEIGHTS)
    evaluate_model(
        model,
        test_dataset.batch(batch_size=BATCH_SIZE),
        N_PARAMS,
        QUANTILES,
    )
