#!/bin/bash
#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=n2c48m24
#SBATCH --job-name=download_data
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4

cd /scratch/ap9283/deep_learning/SSL_project_comp
source /scratch/ap9283/env.sh
conda activate ssl

python download_data.py