#!/bin/bash

#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu
#SBATCH --time=12:00:00
#SBATCH --job-name=simclr_a100
#SBATCH --output=/scratch/ap9283/deep_learning/logs/simclr_%j.out
#SBATCH --error=/scratch/ap9283/deep_learning/logs/simclr_%j.err
#SBATCH --requeue
#SBATCH --exclude=b-9-61

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Initialize conda properly
source /scratch/ap9283/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate ssl

# Navigate to project directory
cd /scratch/ap9283/deep_learning/SSL_project_comp/simplecrl

# Verify we're in the right place
echo "Working directory: $(pwd)"
ls -la

# Run training
python main.py

echo "End time: $(date)"
