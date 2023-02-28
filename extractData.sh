#!/bin/bash
#
#
#SBATCH --partition=gpu
#SBATCH --account=cosc027924

# Change into working directory
cd "${SLURM_SUBMIT_DIR}"

# Hello World program
python dataset_management/redd/create_trainset_redd.py