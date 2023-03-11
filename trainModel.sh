#!/bin/bash
#
#
#SBATCH --partition=gpu
#SBATCH --account=cosc027924

# Change into working directory
cd "${SLURM_SUBMIT_DIR}"

# Hello World program
python train_main.py --appliance_name microwave --crop 1571848 --epochs 100