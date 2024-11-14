from pathlib import Path
import os

from pps_mw_training.models.trainers.utils import AugmentationType
from pps_mw_training.pipelines.cloud_base.input_params import (
    ALL_LABEL_PARAMS,
    ALL_INPUT_PARAMS,
    get_selected_params,
)

MODEL_CONFIG_PATH = Path(os.environ.get("MODEL_CONFIG_CLOUD_BASE", "/tmp"))
TRAINING_DATA_PATH = Path(
    os.environ.get(
        "TRAINING_DATA_PATH_CLOUD_BASE",
        "/tmp",
    )
)


# Select the parameters you want to use as training inputs
SELECTED_TRAINING_INPUT_NAMES = [
    "M05",
    "M07",
    "M12",
    "M15",
    "M16",
    "h_2meter",
    "t_2meter",
    "p_surface",
    "z_surface",
    "ciwv",
    "t250",
    "t400",
    "t500",
    "t700",
    "t850",
    "t900",
    "t950",
    "t1000",
    "t_sea",
    "t_land",
    "q250",
    "q400",
    "q500",
    "q700",
    "q850",
    "q900",
    "q950",
    "q1000",
]
# Select the parameters you want to use as training labels
SELECTED_TRAINING_LABEL_NAMES = "cloud_base"


QUANTILES = [
    0.005,
    0.025,
    0.050,
    0.150,
    0.250,
    0.500,
    0.750,
    0.85,
    0.95,
    0.975,
    0.995,
]

N_UNET_BASE = 16
N_UNET_BLOCKS = 4
N_FEATURES = 32
N_LAYERS = 2

# training parameters
N_EPOCHS = 5
BATCH_SIZE = 128
TRAIN_FRACTION = 0.8
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.05
FILL_VALUE_IMAGES = -999.0
FILL_VALUE_LABELS = -900.0
IMAGE_SIZE = 16
# learning rate parameters
INITIAL_LEARNING_RATE = 0.01
DECAY_STEPS_FACTOR = 0.99
ALPHA = 0.1

AUGMENTATION_TYPE = AugmentationType.CROP_AND_FLIP_CENTERED
SUPER_RESOLUTION = False


INPUT_PARAMS = get_selected_params(
    SELECTED_TRAINING_INPUT_NAMES, ALL_INPUT_PARAMS
)
LABEL_PARAMS = get_selected_params(
    [SELECTED_TRAINING_LABEL_NAMES], ALL_LABEL_PARAMS
)
