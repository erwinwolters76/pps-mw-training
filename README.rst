===============
pps-mw-training
===============
----------------------------------------
a tool for training of pps-mw algorithms
----------------------------------------

Quickstart
==========

Installation
------------

.. code-block:: console

  $ mamba env create -f pps_mw_training_environment.yml

  $ conda activate pps-mw-training

  $ pip install .


Run tests
---------

The package contains a test suite can be run by tox_:

.. code-block:: console 

  $ pip install tox

  $ tox

.. _tox: https://pypi.org/project/tox/

Run package scripts
-------------------

.. code-block:: console

  $ train --help

  usage: train [-h] [-b BATCH_SIZE] [-d DB_FILE] [-e EPOCHS] [-l N_HIDDEN_LAYERS] [-n N_NEURONS_PER_HIDDEN_LAYER] [-o] [-w MODEL_CONFIG_PATH]

  Run the pps-mw training app for training a quantile regression neural network to retrieve ice water path from ICI data.

  options:
    -h, --help            show this help message and exit
    -b BATCH_SIZE, --batchsize BATCH_SIZE
                          Training batch size, default is 4096
    -d DB_FILE, --db-file DB_FILE
                          ICI retrieval database file, default is /home/a002491/ici_retrieval_database.nc
    -e EPOCHS, --epochs EPOCHS
                          Number of epochs, default is 32
    -l N_HIDDEN_LAYERS, --layers N_HIDDEN_LAYERS
                          Number of hidden layers, default is 4
    -n N_NEURONS_PER_HIDDEN_LAYER, --neurons N_NEURONS_PER_HIDDEN_LAYER
                          Number of hidden layers, default is 128
    -o, --only-evaluate   Flag for only evaluating a pretrained model
    -w MODEL_CONFIG_PATH, --write MODEL_CONFIG_PATH
                          Path to use for saving the trained model config, or to read from for an evaluation purpose, default is saved_model
