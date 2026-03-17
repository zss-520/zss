#!/bin/bash
#SBATCH --job-name=stage1_exploration
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err

# Load necessary modules if required by your cluster
# module load python/3.8

# Ensure log directory exists
mkdir -p logs
mkdir -p data

# Change to the directory where the script is submitted
cd $SLURM_SUBMIT_DIR

echo "Starting Stage 1 Exploration at $(date)"

# Run the Python script
# Ensure the python environment has access to necessary system paths
/usr/bin/python3 stage1_exploration.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Stage 1 Pipeline completed successfully at $(date)"
else
    echo "Stage 1 Pipeline failed with exit code $? at $(date)"
    exit 1
fi
