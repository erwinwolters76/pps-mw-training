from pathlib import Path
from typing import Dict, List, Union
import os

from pps_mw_training.models.trainers.utils import AugmentationType

MODEL_CONFIG_PATH = Path(
    os.environ.get("MODEL_CONFIG_CLOUD_BASE", "/home/sm_indka/temp/try/")
)
TRAINING_DATA_PATH = Path(
    os.environ.get(
        "TRAINING_DATA_PATH_CLOUD_BASE",
        "/nobackup/smhid20/users/sm_indka/collocated_data/filtered_data/",
    )
)
# model parameters
INPUT_PARAMS: List[Dict[str, Union[str, float, int]]] = [
    {
        "name": "M01",
        "scale": "linear",
        "min": 9.0,
        "max": 156.92,
        "mean": 45.722798145709,
        "std": 21.95480387639247,
        "zscore_normalise": True,
    },
    {
        "name": "M02",
        "scale": "linear",
        "min": 7.25,
        "max": 156.93,
        "mean": 43.472239755819054,
        "std": 22.867776801356122,
        "zscore_normalise": True,
    },
    {
        "name": "M03",
        "scale": "linear",
        "min": 5.46,
        "max": 160.43,
        "mean": 40.923600920967864,
        "std": 23.409937236096514,
        "zscore_normalise": True,
    },
    {
        "name": "M04",
        "scale": "linear",
        "min": 3.5099998,
        "max": 160.21,
        "mean": 36.45118479368475,
        "std": 22.51684208364647,
        "zscore_normalise": True,
    },
    {
        "name": "M05",
        "scale": "linear",
        "min": 1.85,
        "max": 157.48999,
        "mean": 36.142117032150246,
        "std": 24.38639123384754,
        "zscore_normalise": True,
    },
    {
        "name": "M06",
        "scale": "linear",
        "min": 1.2,
        "max": 159.85999,
        "mean": 21.87554449134835,
        "std": 15.935278292929638,
        "zscore_normalise": True,
    },
    {
        "name": "M07",
        "scale": "linear",
        "min": 0.75,
        "max": 156.33,
        "mean": 38.15007132764771,
        "std": 24.576855357564384,
        "zscore_normalise": True,
    },
    {
        "name": "M08",
        "scale": "linear",
        "min": 0.75,
        "max": 156.33,
        "mean": 38.15007132764771,
        "std": 24.576855357564384,
        "zscore_normalise": True,
    },
    {
        "name": "M09",
        "scale": "linear",
        "min": 0.02,
        "max": 139.29,
        "mean": 4.793895239582563,
        "std": 8.85750747797539,
        "zscore_normalise": True,
    },
    {
        "name": "M10",
        "scale": "linear",
        "min": 0.08,
        "max": 155.22,
        "mean": 21.83183990110206,
        "std": 13.875611051817904,
        "zscore_normalise": True,
    },
    {
        "name": "M11",
        "scale": "linear",
        "min": 0.01,
        "max": 159.42,
        "mean": 19.034346107150558,
        "std": 12.283460305970328,
        "zscore_normalise": True,
    },
    {
        "name": "M12",
        "scale": "linear",
        "min": 204.67420959472656,
        "max": 360.1435852050781,
        "mean": 284.0697882722524,
        "std": 17.423538257835077,
        "zscore_normalise": True,
    },
    {
        "name": "M13",
        "scale": "linear",
        "min": 191.92042541503906,
        "max": 363.63702392578125,
        "mean": 272.0085796736719,
        "std": 17.56035725381957,
        "zscore_normalise": True,
    },
    {
        "name": "M14",
        "scale": "linear",
        "min": 178.8894500732422,
        "max": 326.28753662109375,
        "mean": 266.56287274085213,
        "std": 18.307663499201603,
        "zscore_normalise": True,
    },
    {
        "name": "M15",
        "scale": "linear",
        "min": 177.5694122314453,
        "max": 334.90264892578125,
        "mean": 267.87728514528317,
        "std": 19.500754334747885,
        "zscore_normalise": True,
    },
    {
        "name": "M16",
        "scale": "linear",
        "min": 177.65817260742188,
        "max": 333.8951416015625,
        "mean": 266.6215824795838,
        "std": 19.38803496330258,
        "zscore_normalise": True,
    },
    {
        "name": "h_2meter",
        "scale": "linear",
        "min": 3.5267482,
        "max": 100.0139,
        "mean": 76.53193867314339,
        "std": 16.630991795963556,
        "zscore_normalise": True,
    },
    {
        "name": "t_2meter",
        "scale": "linear",
        "min": 215.69734,
        "max": 321.26932,
        "mean": 285.5732175305176,
        "std": 13.72634376284516,
        "zscore_normalise": True,
    },
    {
        "name": "p_surface",
        "scale": "linear",
        "min": 10000,
        "max": 104861.734,
        "mean": 98391.67473109375,
        "std": 6456.561415968002,
        "zscore_normalise": True,
    },
    {
        "name": "z_surface",
        "scale": "linear",
        "min": 0.0,
        "max": 54079.645,
        "mean": 2433.1688326830313,
        "std": 5949.827141663237,
        "zscore_normalise": True,
    },
    {
        "name": "ciwv",
        "scale": "linear",
        "min": 0.10110732,
        "max": 77.626335,
        "mean": 23.143573866659402,
        "std": 16.47412636069295,
        "zscore_normalise": True,
    },
    {
        "name": "t250",
        "scale": "linear",
        "min": 197.65186,
        "max": 242.42845,
        "mean": 224.69264364868164,
        "std": 6.968061496997991,
        "zscore_normalise": True,
    },
    {
        "name": "t400",
        "scale": "linear",
        "min": 216.0558,
        "max": 265.25317,
        "mean": 245.3769554107666,
        "std": 10.1230745807611,
        "zscore_normalise": True,
    },
    {
        "name": "t500",
        "scale": "linear",
        "min": 222.96696,
        "max": 277.9416,
        "mean": 256.3739975970459,
        "std": 10.107934260577437,
        "zscore_normalise": True,
    },
    {
        "name": "t700",
        "scale": "linear",
        "min": 215.31038,
        "max": 316.67392,
        "mean": 271.5975372644043,
        "std": 10.673382563556011,
        "zscore_normalise": True,
    },
    {
        "name": "t850",
        "scale": "linear",
        "min": 223.40952,
        "max": 333.14856,
        "mean": 279.3380216827393,
        "std": 11.950486262270868,
        "zscore_normalise": True,
    },
    {
        "name": "t900",
        "scale": "linear",
        "min": 225.85133,
        "max": 336.9157,
        "mean": 281.54549735229494,
        "std": 12.608327215777589,
        "zscore_normalise": True,
    },
    {
        "name": "t950",
        "scale": "linear",
        "min": 228.18555,
        "max": 340.8665,
        "mean": 283.7890126626587,
        "std": 13.363535983585342,
        "zscore_normalise": True,
    },
    {
        "name": "t1000",
        "scale": "linear",
        "min": 229.97408,
        "max": 344.208,
        "zscore_normalise": True,
        "mean": 286.7742821728768,
        "std": 14.474404062499847,
    },
    {
        "name": "t_sea",
        "scale": "linear",
        "min": 209.67264,
        "max": 339.43146,
        "zscore_normalise": True,
        "mean": 286.60209551483155,
        "std": 15.19607293352855,
    },
    {
        "name": "t_land",
        "scale": "linear",
        "min": 209.67264,
        "max": 339.43146,
        "zscore_normalise": True,
        "mean": 286.60209551483155,
        "std": 15.19607293352855,
    },
    # {
    #     "name": "rh250",
    #     "scale": "linear",
    #     "min": 8.909693e-07,
    #     "max": 0.015618654,
    #     "zscore_normalise": True,
    #     "mean": 0.005291610806524404,
    #     "std": 0.003723000052433534,
    # },
    # {
    #     "name": "rh400",
    #     "scale": "linear",
    #     "min": 0.0,
    #     "max": 0.014202618,
    #     "zscore_normalise": True,
    #     "mean": 0.006046445369025605,
    #     "std": 0.0033096767106421854,
    # },
    # {
    #     "name": "rh500",
    #     "scale": "linear",
    #     "min": 0.0,
    #     "max": 0.013325257,
    #     "zscore_normalise": True,
    #     "mean": 0.005735704751800622,
    #     "std": 0.003239517962875649,
    # },
    # {
    #     "name": "rh700",
    #     "scale": "linear",
    #     "min": 9.798279e-07,
    #     "max": 0.026251623,
    #     "zscore_normalise": True,
    #     "mean": 0.006034013814882565,
    #     "std": 0.0029108583874336603,
    # },
    # {
    #     "name": "rh850",
    #     "scale": "linear",
    #     "min": 6.039675e-07,
    #     "max": 0.039872587,
    #     "zscore_normalise": True,
    #     "mean": 0.007134334129411873,
    #     "std": 0.0024109007921006044,
    # },
    # {
    #     "name": "rh900",
    #     "scale": "linear",
    #     "min": 3.3620567e-05,
    #     "max": 0.03770326,
    #     "zscore_normalise": True,
    #     "mean": 0.007627256122758554,
    #     "std": 0.0022071383987135465,
    # },
    # {
    #     "name": "rh950",
    #     "scale": "linear",
    #     "min": 0.00015493887,
    #     "max": 0.043908663,
    #     "zscore_normalise": True,
    #     "mean": 0.007926371323379571,
    #     "std": 0.0021499258670604083,
    # },
    # {
    #     "name": "rh1000",
    #     "scale": "linear",
    #     "min": 0.00016252055,
    #     "max": 0.03668715,
    #     "zscore_normalise": True,
    #     "mean": 0.00738849284664931,
    #     "std": 0.002174385255368654,
    # },
    {
        "name": "q250",
        "scale": "linear",
        "min": 1.0018135e-08,
        "max": 0.0007030257,
        "zscore_normalise": True,
        "mean": 6.507605248566506e-05,
        "std": 7.889838169028413e-05,
    },
    {
        "name": "q400",
        "scale": "linear",
        "min": -2.6186394e-06,
        "max": 0.0044494537,
        "zscore_normalise": True,
        "mean": 0.00046727919108313247,
        "std": 0.0005378467546237773,
    },
    {
        "name": "q500",
        "scale": "linear",
        "min": -2.1237513e-06,
        "max": 0.008851501,
        "zscore_normalise": True,
        "mean": 0.001033781902913779,
        "std": 0.0011316728144658357,
    },
    {
        "name": "q700",
        "scale": "linear",
        "min": 4.5555743e-07,
        "max": 0.014707792,
        "zscore_normalise": True,
        "mean": 0.0028668664924348104,
        "std": 0.002547806436435812,
    },
    {
        "name": "q850",
        "scale": "linear",
        "min": 3.7288197e-07,
        "max": 0.021460181,
        "zscore_normalise": True,
        "mean": 0.005026202192237133,
        "std": 0.003941367242371887,
    },
    {
        "name": "q900",
        "scale": "linear",
        "min": 5.4097072e-06,
        "max": 0.021461315,
        "zscore_normalise": True,
        "mean": 0.005886857095257228,
        "std": 0.004439638445114495,
    },
    {
        "name": "q950",
        "scale": "linear",
        "min": 5.4097072e-06,
        "max": 0.021826234,
        "zscore_normalise": True,
        "mean": 0.006777255145888057,
        "std": 0.005117977373985545,
    },
    {
        "name": "q1000",
        "scale": "linear",
        "min": 5.4097072e-06,
        "max": 0.024079349,
        "zscore_normalise": True,
        "mean": 0.007224722144002037,
        "std": 0.0054276352054161625,
    },
]
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

# training labels
TRAINING_LABEL_NAME = "cloud_base"
TRAINING_LABEL_MAX = 12000.0  # m
TRAINING_LABEL_MIN = 0.0  # m
UPDATE_STD_MEAN = False

AUGMENTATION_TYPE = AugmentationType.CROP_AND_FLIP_CENTERED
SUPER_RESOLUTION = False
