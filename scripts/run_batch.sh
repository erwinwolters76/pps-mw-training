#!/bin/bash -l                                                                                                    
#SBATCH -A safnwc
#SBATCH -n 3 # lower case n!
#SBATCH -t 24:65:00 #Maximum time of 7:65 minutes

module load Anaconda/2023.09-0-hpc1
conda activate pps-mw-training


python train.py cloud_base  
#python make_training_data_pixel_based_old_db.py
#python split_files.py
#python mean_std.py
