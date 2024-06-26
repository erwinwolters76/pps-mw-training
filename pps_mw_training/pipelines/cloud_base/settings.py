from pathlib import Path
from typing import Dict, List, Union
import os

from pps_mw_training.models.trainers.utils import AugmentationType

MODEL_CONFIG_PATH = Path(os.environ.get("MODEL_CONFIG_CLOUD_BASE", "/home/sm_indka/temp/"))
TRAINING_DATA_PATH = Path(
    os.environ.get(
        "TRAINING_DATA_PATH_CLOUD_BASE",
        "/nobackup/smhid20/users/sm_indka/collocated_data/"
    )
)
# model parameters
INPUT_PARAMS: List[Dict[str, Union[str, float, int]]] = [
    # {
    #     "name": "M01",
    #     "scale": "linear",
    #     "min": 0.1,
    #     "max": 150.0,
    #     "zscore_normalise": False,
    # },
    # {
    #     "name": "M02",
    #     "scale": "linear",
    #     "min": 0.1,
    #     "max": 150.0,
    #     "zscore_normalise": False,
    # },
    {
        "name": "M03",
        "scale": "linear",
        "min": 0.1,
        "max": 150.0,
        "zscore_normalise": False,
    },
    {
        "name": "M04",
        "scale": "linear",
        "min": 0.1,
        "max": 150.0,
        "zscore_normalise": False,
    },
    # {
    #     "name": "M05",
    #     "scale": "linear",
    #     "min": 0.1,
    #     "max": 150.0,
    #     "zscore_normalise": False,
    # },
    # {
    #     "name": "M06",
    #     "scale": "linear",
    #     "min": 0.1,
    #     "max": 150.0,
    #     "zscore_normalise": False,
    # },
    # {
    #     "name": "M07",
    #     "scale": "linear",
    #     "min": 0.1,
    #     "max": 150.0,
    #     "zscore_normalise": False,
    # },
    {
        "name": "M08",
        "scale": "linear",
        "min": 0.1,
        "max": 150.0,
        "zscore_normalise": False,
    },
    {
        "name": "M09",
        "scale": "linear",
        "min": 0.1,
        "max": 150.0,
        "zscore_normalise": False,
    },
    {
        "name": "M10",
        "scale": "linear",
        "min": 0.1,
        "max": 150.0,
        "zscore_normalise": False,
    },
    # {
    #     "name": "M11",
    #     "scale": "linear",
    #     "min": 0.1,
    #     "max": 150.0,
    #     "zscore_normalise": False,
    # },
    {
        "name": "M12",
        "scale": "linear",
        "min": 175.0,
        "max": 370.0,
        "zscore_normalise": False,
    },
    {
        "name": "M13",
        "scale": "linear",
        "min": 175.0,
        "max": 335.0,
        "zscore_normalise": False,
    },
    # {
    #     "name": "M14",
    #     "scale": "linear",
    #     "min": 175.0,
    #     "max": 335.0,
    #     "zscore_normalise": False,
    # },
    # {
    #     "name": "M15",
    #     "scale": "linear",
    #     "min": 175.0,
    #     "max": 340.0,
    #     "zscore_normalise": False,
    # },
    # {
    #     "name": "M16",
    #     "scale": "linear",
    #     "min": 175.0,
    #     "max": 340.0,
    #     "zscore_normalise": False,
    # },
    {
        "name": "h_2meter",
        "scale": "linear",
        "min": 4.0,
        "max": 100.0,
        "zscore_normalise": False,
    },
    {
        "name": "t_2meter",
        "scale": "linear",
        "min": 200.0,
        "max": 330.0,
        "zscore_normalise": False,
    },
    # {
    #     "name": "p_surface",
    #     "scale": "linear",
    #     "min": 150.0,
    #     "max": 300.0,
    #     "zscore_normalise": False,
    # },
    {
        "name": "z_surface",
        "scale": "linear",
        "min": 0.0,
        "max": 8000.0,
        "zscore_normalise": False,
    },
    {
        "name": "ciwv",
        "scale": "linear",
        "min": 0.1,
        "max": 80.0,
        "zscore_normalise": False,
    },
    # {
    #     "name": "q100",
    #     "scale": "linear",
    #     "min": 150.0,
    #     "max": 300.0,
    #     "zscore_normalise": False,
    # },
    # {
    #     "name": "q250",
    #     "scale": "linear",
    #     "min": 150.0,
    #     "max": 300.0,
    #     "zscore_normalise": False,
    # },
    # {
    #     "name": "q400",
    #     "scale": "linear",
    #     "min": 150.0,
    #     "max": 300.0,
    #     "zscore_normalise": False,
    # },
    # {
    #     "name": "q500",
    #     "scale": "linear",
    #     "min": 150.0,
    #     "max": 300.0,
    #     "zscore_normalise": False,
    # },
    # {
    #     "name": "q700",
    #     "scale": "linear",
    #     "min": 150.0,
    #     "max": 300.0,
    #     "zscore_normalise": False,
    # },
    # {
    #     "name": "q850",
    #     "scale": "linear",
    #     "min": 150.0,
    #     "max": 300.0,
    #     "zscore_normalise": False,
    # },
    # {
    #     "name": "q900",
    #     "scale": "linear",
    #     "min": 150.0,
    #     "max": 300.0,
    #     "zscore_normalise": False,
    # },
    # {
    #     "name": "q950",
    #     "scale": "linear",
    #     "min": 150.0,
    #     "max": 300.0,
    #     "zscore_normalise": False,
    # },
    # {
    #     "name": "q1000",
    #     "scale": "linear",
    #     "min": 150.0,
    #     "max": 300.0,
    #     "zscore_normalise": False,
    # },
    {
        "name": "t100",
        "scale": "linear",
        "min": 180.0,
        "max": 330.0,
        "zscore_normalise": False,
    },
    {
        "name": "t250",
        "scale": "linear",
        "min": 180.0,
        "max": 330.0,
        "zscore_normalise": False,
    },
    {
        "name": "t400",
        "scale": "linear",
        "min": 180.0,
        "max": 330.0,
        "zscore_normalise": False,
    },
    {
        "name": "t500",
        "scale": "linear",
        "min": 180.0,
        "max": 330.0,
        "zscore_normalise": False,
    },
    {
        "name": "t700",
        "scale": "linear",
        "min": 180.0,
        "max": 330.0,
        "zscore_normalise": False,
    },
    {
        "name": "t850",
        "scale": "linear",
        "min": 180.0,
        "max": 330.0,
        "zscore_normalise": False,
    },
    {
        "name": "t900",
        "scale": "linear",
        "min": 180.0,
        "max": 330.0,
        "zscore_normalise": False,
    },
    {
        "name": "t950",
        "scale": "linear",
        "min": 180.0,
        "max": 330.0,
        "zscore_normalise": False,
    },
    {
        "name": "t1000",
        "scale": "linear",
        "min": 180.0,
        "max": 330.0,
        "zscore_normalise": False,
    },
    {
        "name": "t_sea",
        "scale": "linear",
        "min": 180.0,
        "max": 300.0,
        "zscore_normalise": False,
    },
    {
        "name": "t_land",
        "scale": "linear",
        "min": 180.0,
        "max": 330.0,
        "zscore_normalise": False,
    },
]
QUANTILES = [0.005, 0.025, 0.150, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
N_UNET_BASE = 16
N_UNET_BLOCKS = 4
N_FEATURES = 16
N_LAYERS = 4

# training parameters
N_EPOCHS = 20
BATCH_SIZE = 10
TRAIN_FRACTION = 0.8
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.05
FILL_VALUE_IMAGES = -1.5
FILL_VALUE_LABELS = -100.0
IMAGE_SIZE = 32
# learning rate parameters
INITIAL_LEARNING_RATE = 0.01
DECAY_STEPS_FACTOR = 0.7
ALPHA = 0.1

AUGMENTATION_TYPE = AugmentationType.FLIP
SUPER_RESOLUTION = False
