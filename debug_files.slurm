#!/bin/bash
#SBATCH --job-name=debug_files
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=5:00
#SBATCH --output=log/debug-files-%j.out
#SBATCH --error=log/debug-files-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

echo "=== FILE DEBUG $(date) ==="
echo "Working directory: $(pwd)"
echo ""
echo "Looking for container files:"
ls -la *.sif 2>/dev/null || echo "No .sif files found"
echo ""
echo "Looking for model files:"
ls -la *.onnx 2>/dev/null || echo "No .onnx files found"  
echo ""
echo "Looking for verification script:"
ls -la verify*.py 2>/dev/null || echo "No verify*.py files found"
echo ""
echo "All files in directory:"
ls -la
echo ""
echo "=== END FILE DEBUG ==="
