#!/bin/bash
#SBATCH --job-name=debug_test
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=log/debug_%j.out
#SBATCH --error=log/debug_%j.err

# Simple debug script to test SLURM environment
echo "=== SLURM DEBUG TEST ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "Environment variables:"
env | grep SLURM | head -10

echo "=== Testing basic commands ==="
echo "Python version:"
python --version 2>&1 || echo "Python not found"

echo "Docker availability:"
which docker 2>&1 || echo "Docker not in PATH"
docker --version 2>&1 || echo "Docker command failed"

echo "Available modules:"
module avail 2>&1 | head -10 || echo "Module command not available"

echo "Files in current directory:"
ls -la

echo "=== Test completed ==="
sleep 10  # Keep job running for a bit
echo "Job ending at: $(date)"
