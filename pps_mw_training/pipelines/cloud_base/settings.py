from pathlib import Path
from typing import Dict, List, Union
import os


MODEL_CONFIG_PATH = Path(os.environ.get("MODEL_CONFIG_PR_NORDIC", "/tmp"))
TRAINING_DATA_PATH = Path(os.environ.get("TRAINING_DATA_PATH_PR_NORDIC", "/tmp"))
# model parameters
INPUT_PARAMS: List[Dict[str, Union[str, float, int]]] = [
    {
        "name": "M01",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "M02",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "M03",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "M04",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "M05",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "M06",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "M07",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "M08",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "M09",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "h_2meter",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t_2meter",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "p_surface",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "z_surface",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "ciwv",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "q100",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "q250",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "q400",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "q500",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "q700",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "q850",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "q900",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "q950",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "q1000",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t100",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t250",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t400",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t500",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t700",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t850",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t900",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t950",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t1000",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t100",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t250",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t400",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t500",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t700",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t850",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t900",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t_sea",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
    {
        "name": "t_land",
        "scale": "linear",
        "min": 150.0,
        "max": 300.0,
        "zscore_normalise": True,
    },
]
QUANTILES = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
N_UNET_BASE = 16
N_UNET_BLOCKS = 4
N_FEATURES = 128
N_LAYERS = 4

# training parameters
N_EPOCHS = 32
BATCH_SIZE = 20
TRAIN_FRACTION = 0.8
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.05
FILL_VALUE_IMAGES = -1.5
FILL_VALUE_LABELS = -100.0
IMAGE_SIZE = 32
# learning rate parameters
INITIAL_LEARNING_RATE = 0.0005
DECAY_STEPS_FACTOR = 0.7
ALPHA = 0.1
