#!/usr/bin/env python
import argparse
from pathlib import Path
from sys import argv
from typing import Optional

from pps_mw_training.pipelines.iwp_ici import training as iit
from pps_mw_training.pipelines.pr_nordic import training as pnt
from pps_mw_training.pipelines.pipeline_type import PipelineType


def add_parser(
    subparsers: argparse._SubParsersAction,
    pipeline_type: PipelineType,
    description: str,
    n_hidden_layers: int,
    n_neurons_per_hidden_layer: int,
    batchsize: int,
    n_epochs: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    model_config_path: Path,
    missing_fraction: Optional[float] = None,
    activation: Optional[str] = None,
    db_file: Optional[Path] = None,
    training_data_path: Optional[Path] = None,
):
    """Add parser and set default values."""
    parser = subparsers.add_parser(
        pipeline_type.value,
        description=description,
        help=description,
    )
    if activation is not None:
        parser.add_argument(
            "-a",
            "--activation",
            dest="activation",
            type=str,
            help=(
                "Activation function to use for the hidden layers, "
                f"default is {activation}"
            ),
            default=activation,
        )
    parser.add_argument(
        "-b",
        "--batchsize",
        dest="batch_size",
        type=int,
        help=(
            "Training batch size, "
            f"default is {batchsize}"
        ),
        default=batchsize,
    )
    if db_file is not None:
        parser.add_argument(
            "-d",
            "--db-file",
            dest="db_file",
            type=str,
            help=(
                "Path to ICI retrieval database file to use as training data, "
                f"default is {db_file.as_posix()}"
            ),
            default=db_file.as_posix(),
        )

    parser.add_argument(
        "-e",
        "--epochs",
        dest="n_epochs",
        type=int,
        help=(
            "Number of training epochs, "
            f"default is {n_epochs}"
        ),
        default=n_epochs,
    )
    parser.add_argument(
        "-l",
        "--layers",
        dest="n_hidden_layers",
        type=int,
        help=(
            "Number of hidden layers, "
            f"default is {n_hidden_layers}"
        ),
        default=n_hidden_layers,
    )
    if missing_fraction is not None:
        parser.add_argument(
            "-m",
            "--missing-fraction",
            dest="missing_fraction",
            type=float,
            help=(
                "Set this fraction of observations to a fill value, "
                "in order to allow for the network to learn to handle "
                f"missing data, default is {missing_fraction}"
            ),
            default=missing_fraction,
        )
    parser.add_argument(
        "-n",
        "--neurons",
        dest="n_neurons_per_hidden_layer",
        type=int,
        help=(
            "Number of hidden layers, "
            f"default is {n_neurons_per_hidden_layer}"
        ),
        default=n_neurons_per_hidden_layer,
    )
    parser.add_argument(
        "-o",
        "--only-evaluate",
        dest="only_evaluate",
        action="store_true",
        help="Flag for only evaluating a pretrained model",
    )
    if training_data_path is not None:
        parser.add_argument(
            "-p",
            "--training-datapath",
            dest="training_data_path",
            type=str,
            help=(
                "Path to training data, "
                f"default is {training_data_path.as_posix()}"
            ),
            default=training_data_path.as_posix(),
        )
    parser.add_argument(
        "-t",
        "--train-fraction",
        dest="train_fraction",
        type=float,
        help=(
            "Fraction of the training dataset to use as training data, "
            f"default is {train_fraction}"
        ),
        default=train_fraction,
    )
    parser.add_argument(
        "-u",
        "--test-fraction",
        dest="test_fraction",
        type=float,
        help=(
            "Fraction of the training dataset to use as test data, "
            f"default is {test_fraction}"
        ),
        default=test_fraction,
    )
    parser.add_argument(
        "-v",
        "--validation-fraction",
        dest="validation_fraction",
        type=float,
        help=(
            "Fraction of the training dataset to use as validation data, "
            f"default is {validation_fraction}"
        ),
        default=validation_fraction,
    )
    parser.add_argument(
        "-w",
        "--write",
        dest="model_config_path",
        type=str,
        help=(
            "Path to use for saving the trained model config, "
            "or to read from for an evaluation purpose, "
            f"default is {model_config_path.as_posix()}"
        ),
        default=model_config_path.as_posix(),
    )


def cli(args_list: list[str] = argv[1:]) -> None:
    parser = argparse.ArgumentParser(
        description="""Run the pps-mw-training app."""
    )
    subparsers = parser.add_subparsers(dest='pipeline_type')
    add_parser(
        subparsers,
        PipelineType.PR_NORDIC,
        (
            "Run the Nordic precip training pipeline. "
            "N.B. this pipeline is not yet fully implemented."
        ),
        pnt.settings.N_LAYERS,
        pnt.settings.N_FEATURES,
        pnt.settings.BATCH_SIZE,
        pnt.settings.N_EPOCHS,
        pnt.settings.TRAIN_FRACTION,
        pnt.settings.VALIDATION_FRACTION,
        pnt.settings.TEST_FRACTION,
        pnt.settings.MODEL_CONFIG_PATH,
        training_data_path=pnt.settings.TRAINING_DATA_PATH,
    )
    add_parser(
        subparsers,
        PipelineType.IWP_ICI,
        (
            "Run the pps-mw training app for the training of a single "
            "quantile regression neural network, handling multiple "
            "quantiles and retrieval parameters, and missing data, "
            "to retrieve ice water path and other associated parameters "
            "from ICI data."
        ),
        iit.settings.N_HIDDEN_LAYERS,
        iit.settings.N_NEURONS_PER_HIDDEN_LAYER,
        iit.settings.BATCH_SIZE,
        iit.settings.N_EPOCHS,
        iit.settings.TRAIN_FRACTION,
        iit.settings.VALIDATION_FRACTION,
        iit.settings.TEST_FRACTION,
        iit.settings.MODEL_CONFIG_PATH,
        activation=iit.settings.ACTIVATION,
        missing_fraction=iit.settings.MISSING_FRACTION,
        db_file=iit.settings.ICI_RETRIEVAL_DB_FILE,
    )
    args = parser.parse_args(args_list)
    pipeline_type = PipelineType(args.pipeline_type)
    if pipeline_type is PipelineType.PR_NORDIC:
        pnt.train(
            args.n_hidden_layers,
            args.n_neurons_per_hidden_layer,
            Path(args.training_data_path),
            args.train_fraction,
            args.validation_fraction,
            args.test_fraction,
            args.batch_size,
            args.n_epochs,
            Path(args.model_config_path),
            args.only_evaluate,
        )
    else:
        iit.train(
            args.n_hidden_layers,
            args.n_neurons_per_hidden_layer,
            args.activation,
            Path(args.db_file),
            args.train_fraction,
            args.validation_fraction,
            args.test_fraction,
            args.batch_size,
            args.n_epochs,
            args.missing_fraction,
            Path(args.model_config_path),
            args.only_evaluate,
        )


if __name__ == "__main__":
    cli(argv[1:])
