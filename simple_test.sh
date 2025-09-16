#!/bin/bash
#SBATCH --job-name=simple_test
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=log/simple_%j.out
#SBATCH --error=log/simple_%j.err

# Very simple test - no strict error handling
echo "=== SIMPLE SLURM TEST (No strict mode) ==="
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"

echo "Testing module system..."
module purge || echo "Module purge failed but continuing"
module avail 2>&1 | head -5 || echo "Module avail failed but continuing"

echo "Testing basic commands..."
python --version || echo "Python not found"
which apptainer || echo "Apptainer not found"
which singularity || echo "Singularity not found"

echo "Listing files..."
ls -la

echo "Sleeping for 30 seconds..."
sleep 30

echo "Job completed at: $(date)"
echo "=== SIMPLE TEST FINISHED ==="
