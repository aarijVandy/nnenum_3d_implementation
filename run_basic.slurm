#!/bin/bash
#SBATCH --job-name=test_echo
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err

# Create log directory if it doesn’t exist
mkdir -p log

echo "Job started at $(date) on node $(hostname)"
sleep 30
echo "Hello from SLURM test job after 30 seconds!"
echo "Job finished at $(date)"
