#!/bin/bash

#SBATCH --mail-user=krsiegall@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J MODEL_TRAINING
#SBATCH --output=/home/krsiegall/Terrawarden/turing_logs/MODEL_TRAINING%j.out
#SBATCH --error=/home/krsiegall/Terrawarden/turing_logs/MODEL_TRAINING%j.err

#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH -C A30
#SBATCH -p academic
#SBATCH -t 48:00:00

module load python
module load cuda
source ~/bin/activate
python train.py
