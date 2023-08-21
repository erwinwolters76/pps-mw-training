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

The package contains three types of scripts:

  * `regrid` - for regridding of data to produce a training dataset

  * `reformat` - for reformating of data files to produce a training dataset

  * `train` - for training of neural networks
  
So far the package contains two training pipelines:
  
   * one pipeline (`iwp_ici`) for the training of a multilayer perceptron quantile
     regression neural network for the retrieval of ice water path and
     related quantities from EPS-SG ICI data.
     The training utilizes a simulated training dataset, consisting of a set of
     geophysical states and associated simulated ICI observation, and this dataset
     was generated within an EUMETSAT funded activity. 

   * a second pipeline (`pr_nordic`) for the training of a U-Net convolutional quantile
     regression neural network for the retrieval of precipitation rate
     over the nordic region and from JPSS1/ATMS and N-SPP/ATMS data.
     To run the full training pipeline one must first run the
     `regrid` and `reformat` script to crate the training dataset
     consisting of BALTRAD composites and regridded ATMS data. 
     The full training pipeline is under development.

.. raw:: pdf

    PageBreak

You can run a training pipeline as described below, :


.. code-block:: console

  $ train --help

  usage: train [-h] {pr_nordic,iwp_ici} ...

  Run the pps-mw-training app.

  positional arguments:
    {pr_nordic,iwp_ici}
      pr_nordic          Run the Nordic precip training pipeline.
      iwp_ici            Run the IWP ICI training pipeline.

  options:
    -h, --help           show this help message and exit


.. raw:: pdf

    PageBreak


.. code-block:: console

  $ train iwp_ici --help

  usage: train [-h] [-a ACTIVATION] [-b BATCH_SIZE] [-d DB_FILE] [-e EPOCHS] [-l N_HIDDEN_LAYERS] [-m MISSING_FRACTION] [-n N_NEURONS_PER_HIDDEN_LAYER]
                 [-o] [-t TRAIN_FRACTION] [-u TEST_FRACTION] [-v VALIDATION_FRACTION] [-w MODEL_CONFIG_PATH]

  Run the pps-mw training app for the training of a single quantile regression neural network, handling multiple quantiles and retrieval parameters, and
  missing data, to retrieve ice water path and other associated parameters from ICI data.

  options:
    -h, --help            show this help message and exit
    -a ACTIVATION, --activation ACTIVATION
                          Activation function to use for the hidden layers, default is relu
    -b BATCH_SIZE, --batchsize BATCH_SIZE
                          Training batch size, default is 4096
    -d DB_FILE, --db-file DB_FILE
                          Path to ICI retrieval database file to use as training data, default is /home/a002491/ici_retrieval_database.nc
    -e EPOCHS, --epochs EPOCHS
                          Number of training epochs, default is 256
    -l N_HIDDEN_LAYERS, --layers N_HIDDEN_LAYERS
                          Number of hidden layers, default is 4
    -m MISSING_FRACTION, --missing-fraction MISSING_FRACTION
                          Set this fraction of observations to a fill value, in order to allow for the network to learn to handle missing data, default is 0.1
    -n N_NEURONS_PER_HIDDEN_LAYER, --neurons N_NEURONS_PER_HIDDEN_LAYER
                          Number of hidden layers, default is 128
    -o, --only-evaluate   Flag for only evaluating a pretrained model
    -t TRAIN_FRACTION, --train-fraction TRAIN_FRACTION
                          Fraction of the training dataset to use as training data, default is 0.7
    -u TEST_FRACTION, --test-fraction TEST_FRACTION
                          Fraction of the training dataset to use as test data, default is 0.15
    -v VALIDATION_FRACTION, --validation-fraction VALIDATION_FRACTION
                          Fraction of the training dataset to use as validation data, default is 0.15
    -w MODEL_CONFIG_PATH, --write MODEL_CONFIG_PATH
                          Path to use for saving the trained model config, or to read from for an evaluation purpose, default is /home/a002491/work/pps-mw-
                          training/saved_model
