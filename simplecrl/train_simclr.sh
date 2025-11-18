#!/bin/bash

#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --job-name=simclr_train
#SBATCH --output=/scratch/ap9283/deep_learning/logs/simclr_%j.out
#SBATCH --error=/scratch/ap9283/deep_learning/logs/simclr_%j.err
#SBATCH --requeue

source /scratch/ap9283/env.sh
conda activate ssl

cd /scratch/ap9283/deep_learning/SSL_project_comp/simplecrl

python main.py

