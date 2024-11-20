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


Training pipelines
------------------

The package currently contains three training "pipelines":
  
   * `iwp_ici` - for the training of a multilayer perceptron quantile
     regression neural network for the retrieval of ice water path and
     related quantities from EPS-SG ICI data,

   * `pr_nordic` - for the training of a U-Net convolutional quantile
     regression neural network for the retrieval of precipitation rate
     over the nordic region and from JPSS1/ATMS and N-SPP/ATMS data.

   * `cloud_base` - for the training of a U-Net convolutional quantile
     regression neural network for the retrieval of cloud base height or 
     similar cloud parameters from Cloudsat/Calipso and N-SPP/VIIRS data.     


The three pipelines differ a bit in complexity. The `iwp_ici` pipeline
is less complex and uses a fixed training dataset, and this package
does not contain the tools to generate this training dataset. Similar 
is for `cloud_base` pipeline. However, tools for the generation of a 
training dataset are included for the `pr_nordic` pipeline.

Run package scripts
-------------------

The package contains three scripts:

  * `regrid` - for regridding of data to produce a training dataset,

  * `reformat` - for reformating of data files to produce a training dataset,

  * `train` - for training of neural networks.

You only should run the `train` script for the `iwp_ici` and `cloud_base` pipelines,
but you need to run all three type of scripts for the `pr_nordic` pipeline.
Each script has a command line interface, and you can e.g. run the `train`
script as described below:

.. raw:: pdf

    PageBreak

.. code-block:: console

  $ train --help

  usage: train [-h] {pr_nordic,iwp_ici} ...

  Run the pps-mw-training app.

  positional arguments:
    {pr_nordic,iwp_ici}
      pr_nordic          Run the Nordic precip training pipeline.
      iwp_ici            Run the IWP ICI training pipeline.
      cloud_base         Run the cloud base pipeline

  options:
    -h, --help           show this help message and exit


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

   

Prior to running the actual training of the `pr_nordic` pipeline one must create
the training dataset, i.e: 

  * `regrid` - regrid ATMS data onto positions of the grid of the BALTRAD data,
    note that this is a quite computational expensive operation,

  * `reformat` - reformat BALTRAD data, saves the BALTRAD composites
    in a new file format only including the data needed for the training.
    This step, in practise, only reduces the file size of the BALTRAD composites. 
    Currently, the `train` script only handles reformatted  BALTRAD
    composites, so, at least for now, it is necessary to perform this processing.


Setting environment variables
-----------------------------

You can set the path to the environment variables 
`MODEL_CONFIG_CLOUD_BASE`  -  directory for saved model
`TRAINING_DATA_PATH_CLOUD_BASE`  - directory containing training data

.. code-block:: console
  export MODEL_CONFIG_CLOUD_BASE="/path/to/model/config"
  export TRAINING_DATA_PATH_CLOUD_BASE="/path/to/training/data"
