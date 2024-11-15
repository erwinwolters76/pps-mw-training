#!/bin/bash -l                                                                                                    
#SBATCH -A safnwc
#SBATCH -n 8 # lower case n!
#SBATCH -t 4:00:00 #Maximum time of 4:00 hrs

module load Mambaforge/23.3.1-1-hpc1 
conda activate pps-mw-training

pipeline=cloud_base
python train.py $pipeline
