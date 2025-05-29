#!/bin/bash

### TC1 Job Script ###
 
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

### Specify Memory allocate to this job ###
#SBATCH --mem=16G

### Specify number of core (CPU) to allocate to per node ###
#SBATCH --ntasks-per-node=4

### Specify number of node to compute ###
#SBATCH --nodes=1

### Optional: Specify node to execute the job ###
### Remove 1st # at next line for the option to take effect ###
##SBATCH --nodelist=TC1N06

### Specify Time Limit, format: <min> or <min>:<sec> or <hr>:<min>:<sec> or <days>-<hr>:<min>:<sec> or <days>-<hr> ### 
#SBATCH --time=06:00:00

### Specify name for the job, filename format for output and error ###
#SBATCH --job-name=DS
#SBATCH --output=Z_%j.out
#SBATCH --error=Z_%j.err

### Your script for computation ###
source activate base
conda activate handover
module load cuda/12.2

#export PATH="/home/FYP/dlee055/.conda/envs/daisee3/bin:$PATH"

#find $CONDA_PREFIX -name "libstdc++.so.6"
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
python track_with_sort.py
python processing.py