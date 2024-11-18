#!/bin/bash

#SBATCH --mail-user=krsiegall@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J DOWNLOAD_DATASET
#SBATCH --output=/home/krsiegall/Terrawarden/turing_logs/DOWNLOAD_DATASET%j.out
#SBATCH --error=/home/krsiegall/Terrawarden/turing_logs/DOWNLOAD_DATASET%j.err

#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -p academic
#SBATCH -t 48:00:00

module load python
source ~/bin/activate
python train.py
