#!/usr/bin/env python
from pathlib import Path

from tensorflow.keras.callbacks import ModelCheckpoint

from pps_mw_training import ici
from pps_mw_training.evaluation import evaluate_model
from pps_mw_training.model import QuantileModel
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
# learning rate parameters
INITIAL_LEARNING_RATE = 0.0001
FIRST_DECAY_STEPS = 1000
T_MUL = 2.0
M_MUL = 1.0
ALPHA = 0.0
MODEL_CONFIG_PATH = Path("saved_model/network_config.json")
INPUT_PARAMS = [
    {
        "name": "DTB_ICI_DB_ICI_01V",
        "scale": "linear",
        "min": -170.,
        "max": 30.,
    },
    {
        "name": "DTB_ICI_DB_ICI_02V",
        "scale": "linear",
        "min": -155.,
        "max": 35.,
    },
    {
        "name": "DTB_ICI_DB_ICI_03V",
        "scale": "linear",
        "min": -145.,
        "max": 35.,
    },
    {
        "name": "DTB_ICI_DB_ICI_04V",
        "scale": "linear",
        "min": -195.,
        "max": 40.,
    },
    {
        "name": "DTB_ICI_DB_ICI_04H",
        "scale": "linear",
        "min": -195.,
        "max": 50.,
    },
    {
        "name": "DTB_ICI_DB_ICI_05V",
        "scale": "linear",
        "min": -185.,
        "max": 30.,
    },
    {
        "name": "DTB_ICI_DB_ICI_06V",
        "scale": "linear",
        "min": -180.,
        "max": 30.,
    },
    {
        "name": "DTB_ICI_DB_ICI_07V",
        "scale": "linear",
        "min": -165.,
        "max": 35.,
    },
    {
        "name": "DTB_ICI_DB_ICI_08V",
        "scale": "linear",
        "min": -165.,
        "max": 35.,
    },
    {
        "name": "DTB_ICI_DB_ICI_09V",
        "scale": "linear",
        "min": -155.,
        "max": 35.,
    },
    {
        "name": "DTB_ICI_DB_ICI_10V",
        "scale": "linear",
        "min": -135.,
        "max": 25.,
    },
    {
        "name": "DTB_ICI_DB_ICI_11V",
        "scale": "linear",
        "min": -160.,
        "max": 30.,
    },
    {
        "name": "DTB_ICI_DB_ICI_11H",
        "scale": "linear",
        "min": -160.,
        "max": 30.,
    },
    {
        "name": "SurfType",
        "scale": "linear",
        "min": 0.,
        "max": 15.,
    },
    {
        "name": "SurfPres",
        "scale": "linear",
        "min": 50000.,
        "max": 106000.,
    },
    {
        "name": "SurfTemp",
        "scale": "linear",
        "min": 210.,
        "max": 320.,
    },
    {
        "name": "SurfWind",
        "scale": "linear",
        "min": 0.,
        "max": 35.,
    }
]
OUTPUT_PARAMS = [
    {
        "name": "TCWV",
        "scale": "linear",
        "min": 0.,
        "max": 80.,
    },
    {
        "name": "LWP",
        "scale": "linear",
        "min": 0.,
        "max": 2.,
    },
    {
        "name": "RWP",
        "scale": "linear",
        "min": 0.,
        "max": 4.,
    },
    {
        "name": "IWP",
        "scale": "linear",
        "min": 0.,
        "max": 35.,
    },
    {
        "name": "Zmean",
        "scale": "linear",
        "min": 0.,
        "max": 19000.,
    },
    {
        "name": "Dmean",
        "scale": "linear",
        "min": 0.,
        "max": 0.0017,
    }
]


if __name__ == "__main__":
    # Train a quantile regression neural network with simulated ICI data
    TRAIN = True
    FULL_DATASET = ici.load_retrieval_database()
    TRAINING_DATA, TEST_DATA, VALIDATION_DATA = split_dataset(
        FULL_DATASET,
        TRAIN_FRACTION,
        VALIDATION_FRACTION,
        TEST_FRACTION,
    )
    if TRAIN:
        QuantileModel.train(
            INPUT_PARAMS,
            OUTPUT_PARAMS,
            N_HIDDEN_LAYERS,
            N_NEURONS,
            ACTIVATION,
            QUANTILES,
            TRAINING_DATA,
            VALIDATION_DATA,
            BATCH_SIZE,
            EPOCHS,
            INITIAL_LEARNING_RATE,
            FIRST_DECAY_STEPS,
            T_MUL,
            M_MUL,
            ALPHA,
            MODEL_CONFIG_PATH,
        )
    MODEL = QuantileModel.load(MODEL_CONFIG_PATH)
    evaluate_model(MODEL, TEST_DATA)
