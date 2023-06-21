#!/usr/bin/env python
from pathlib import Path
from sys import argv
from typing import Dict, List, Tuple, Union, cast
import argparse
import os

from xarray import Dataset  # type: ignore

from pps_mw_training import ici
from pps_mw_training.quantile_model_evaluation import evaluate_model
from pps_mw_training.quantile_model import QuantileModel
from pps_mw_training.utils import split_dataset, add_noise


ICI_RETRIEVAL_DB_FILE = Path(
    os.environ.get(
        "ICI_RETRIEVAL_DB_FILE", "/tmp/ici_retrieval_database.nc",
    )
)


# model parameters
QUANTILES = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
N_HIDDEN_LAYERS = 4
N_NEURONS_PER_HIDDEN_LAYER = 128
ACTIVATION = "relu"
# training parameters
BATCH_SIZE = 4096
EPOCHS = 256
TRAIN_FRACTION = 0.7
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
NOISE = 1.0
FILL_VALUE = -2.
MISSING_FRACTION = 0.1
# learning rate parameters
INITIAL_LEARNING_RATE = 0.0001
FIRST_DECAY_STEPS = 1000
T_MUL = 2.0
M_MUL = 1.0
ALPHA = 0.0
MODEL_CONFIG_PATH = Path("saved_model").resolve()
INPUT_PARAMS: List[Dict[str, Union[str, float]]] = [
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
OUTPUT_PARAMS: List[Dict[str, Union[str, float]]] = [
    {
        "name": "TCWV",
        "scale": "log",
        "min": 0.,
        "max": 80.,
    },
    {
        "name": "LWP",
        "scale": "log",
        "min": 0.,
        "max": 2.,
    },
    {
        "name": "RWP",
        "scale": "log",
        "min": 0.,
        "max": 4.,
    },
    {
        "name": "IWP",
        "scale": "log",
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


def get_training_data(
    ici_db_file: Path,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Get training data."""
    full_dataset = ici.load_retrieval_database(ici_db_file)
    params = [cast(str, p["name"]) for p in INPUT_PARAMS]
    full_dataset = add_noise(
        full_dataset,
        params=[p for p in params if p.startswith("DTB")],
        sigma=NOISE,
    )
    return split_dataset(
        full_dataset,
        train_fraction,
        validation_fraction,
        test_fraction,
    )


def cli(args_list: List[str] = argv[1:]) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pps-mw training app for the training of a single quantile "
            "regression neural network, handling multiple quantiles and "
            "retrieval parameters, and missing data, to retrieve ice water "
            "path and other associated parameters from ICI data."
        )
    )
    parser.add_argument(
        "-a",
        "--activation",
        dest="activation",
        type=str,
        help=(
            "Activation function to use for the hidden layers, "
            f"default is {ACTIVATION}"
        ),
        default=ACTIVATION,
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        dest="batch_size",
        type=int,
        help=(
            "Training batch size, "
            f"default is {BATCH_SIZE}"
        ),
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "-d",
        "--db-file",
        dest="db_file",
        type=str,
        help=(
            "Path to ICI retrieval database file to use as training data, "
            f"default is {ICI_RETRIEVAL_DB_FILE}"
        ),
        default=ICI_RETRIEVAL_DB_FILE,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        dest="epochs",
        type=int,
        help=(
            "Number of training epochs, "
            f"default is {EPOCHS}"
        ),
        default=EPOCHS,
    )
    parser.add_argument(
        "-l",
        "--layers",
        dest="n_hidden_layers",
        type=int,
        help=(
            "Number of hidden layers, "
            f"default is {N_HIDDEN_LAYERS}"
        ),
        default=N_HIDDEN_LAYERS,
    )
    parser.add_argument(
        "-m",
        "--missing-fraction",
        dest="missing_fraction",
        type=float,
        help=(
            "Set this fraction of observations to a fill value, "
            "in order to allow for the network to learn to handle "
            "missing data, "
            f"default is {MISSING_FRACTION}"
        ),
        default=MISSING_FRACTION,
    )
    parser.add_argument(
        "-n",
        "--neurons",
        dest="n_neurons_per_hidden_layer",
        type=int,
        help=(
            "Number of hidden layers, "
            f"default is {N_NEURONS_PER_HIDDEN_LAYER}"
        ),
        default=N_NEURONS_PER_HIDDEN_LAYER,
    )
    parser.add_argument(
        "-o",
        "--only-evaluate",
        dest="only_evaluate",
        action="store_true",
        help="Flag for only evaluating a pretrained model",
    )
    parser.add_argument(
        "-t",
        "--train-fraction",
        dest="train_fraction",
        type=float,
        help=(
            "Fraction of the training dataset to use as training data, "
            f"default is {TRAIN_FRACTION}"
        ),
        default=TRAIN_FRACTION,
    )
    parser.add_argument(
        "-u",
        "--test-fraction",
        dest="test_fraction",
        type=float,
        help=(
            "Fraction of the training dataset to use as test data, "
            f"default is {TEST_FRACTION}"
        ),
        default=TEST_FRACTION,
    )
    parser.add_argument(
        "-v",
        "--validation-fraction",
        dest="validation_fraction",
        type=float,
        help=(
            "Fraction of the training dataset to use as validation data, "
            f"default is {VALIDATION_FRACTION}"
        ),
        default=VALIDATION_FRACTION,
    )
    parser.add_argument(
        "-w",
        "--write",
        dest="model_config_path",
        type=str,
        help=(
            "Path to use for saving the trained model config, "
            "or to read from for an evaluation purpose, "
            f"default is {MODEL_CONFIG_PATH}"
        ),
        default=MODEL_CONFIG_PATH,
    )
    args = parser.parse_args(args_list)
    db_file = Path(args.db_file)
    train_fraction = args.train_fraction
    validation_fraction = args.validation_fraction
    test_fraction = args.test_fraction
    epochs = args.epochs
    batch_size = args.batch_size
    n_hidden_layers = args.n_hidden_layers
    n_neurons_per_hidden_layer = args.n_neurons_per_hidden_layer
    activation = args.activation
    missing_fraction = args.missing_fraction
    run_training = not args.only_evaluate
    model_config_path = Path(args.model_config_path)
    training_data, test_data, validation_data = get_training_data(
        db_file,
        train_fraction,
        validation_fraction,
        test_fraction,
    )
    if run_training:
        QuantileModel.train(
            INPUT_PARAMS,
            OUTPUT_PARAMS,
            n_hidden_layers,
            n_neurons_per_hidden_layer,
            activation,
            QUANTILES,
            training_data,
            validation_data,
            batch_size,
            epochs,
            INITIAL_LEARNING_RATE,
            FIRST_DECAY_STEPS,
            T_MUL,
            M_MUL,
            ALPHA,
            missing_fraction,
            FILL_VALUE,
            model_config_path,
        )
    model = QuantileModel.load(model_config_path / "network_config.json")
    evaluate_model(model, test_data, missing_fraction, model_config_path)


if __name__ == "__main__":
    cli(argv[1:])
