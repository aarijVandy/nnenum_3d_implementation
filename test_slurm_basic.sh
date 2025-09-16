#!/bin/bash
#SBATCH --job-name=test_basic
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=log/test_basic_%j.out
#SBATCH --error=log/test_basic_%j.err

# Simple test to verify SLURM is working
echo "=== BASIC SLURM TEST ==="
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"

echo "Sleeping for 30 seconds to keep job active..."
sleep 30

echo "Job completed at: $(date)"
echo "=== TEST FINISHED ==="
