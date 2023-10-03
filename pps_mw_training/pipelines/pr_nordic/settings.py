from pathlib import Path
from typing import Dict, List, Union
import os


MODEL_CONFIG_PATH = Path(
    os.environ.get("MODEL_CONFIG_PR_NORDIC", "/tmp")
)
TRAINING_DATA_PATH = Path(
    os.environ.get("TRAINING_DATA_PATH_PR_NORDIC", "/tmp")
)
# model parameters
INPUT_PARAMS: List[Dict[str, Union[str, float, int]]] = [
    {
        "band": "mw_50",
        "index": 0,
        "scale": "linear",
        "min": 150.,
        "max": 300.,
    },
    {
        "band": "mw_50",
        "index": 2,
        "scale": "linear",
        "min": 150.,
        "max": 300.,
    },
    {
        "band": "mw_50",
        "index": 3,
        "scale": "linear",
        "min": 150.,
        "max": 300.,
    },
    {
        "band": "mw_50",
        "index": 4,
        "scale": "linear",
        "min": 150.,
        "max": 300.,
    },
    {
        "band": "mw_50",
        "index": 5,
        "scale": "linear",
        "min": 150.,
        "max": 300.,
    },
    {
        "band": "mw_50",
        "index": 6,
        "scale": "linear",
        "min": 150.,
        "max": 300.,
    },
    {
        "band": "mw_50",
        "index": 7,
        "scale": "linear",
        "min": 150.,
        "max": 300.,
    },
    {
        "band": "mw_90",
        "index": 0,
        "scale": "linear",
        "min": 150.,
        "max": 300.,
    },
    {
        "band": "mw_160",
        "index": 0,
        "scale": "linear",
        "min": 150.,
        "max": 300.,
    },
    {
        "band": "mw_183",
        "index": 0,
        "scale": "linear",
        "min": 190.,
        "max": 290.,
    },
    {
        "band": "mw_183",
        "index": 1,
        "scale": "linear",
        "min": 190.,
        "max": 290.,
    },
    {
        "band": "mw_183",
        "index": 2,
        "scale": "linear",
        "min": 190.,
        "max": 290.,
    },
    {
        "band": "mw_183",
        "index": 3,
        "scale": "linear",
        "min": 190.,
        "max": 290.,
    },
    {
        "band": "mw_183",
        "index": 4,
        "scale": "linear",
        "min": 190.,
        "max": 290.,
    },
]
QUANTILES = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
N_UNET_BASE = 16
N_UNET_BLOCKS = 4
N_FEATURES = 128
N_LAYERS = 4
# radar quality parameters
MIN_QUALITY = 0.8  # radar quality index between 0 (poor) and 1 (good)
MAX_DISTANCE = 200e3  # max distance [m] from radar
# training parameters
N_EPOCHS = 32
BATCH_SIZE = 20
TRAIN_FRACTION = 0.8
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.05
FILL_VALUE_IMAGES = -1.5
FILL_VALUE_LABELS = -100.0
IMAGE_SIZE = 64
# learning rate parameters
INITIAL_LEARNING_RATE = 0.0005
DECAY_STEPS_FACTOR = 0.7
ALPHA = 0.1
