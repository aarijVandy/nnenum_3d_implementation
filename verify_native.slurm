#!/bin/bash
#SBATCH --job-name=verify_native
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=verify-native-%j.out
#SBATCH --error=verify-native-%j.err

# CRITICAL: Set ACCRE variables BEFORE strict mode
export USER_IS_ROOT="${USER_IS_ROOT:-0}"

# Initialize ACCRE software stack early
if type setup_accre_software_stack &>/dev/null; then
    setup_accre_software_stack || true
fi

# NOW enable strict mode
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

echo "=== NATIVE PYTHON VERIFICATION $(date) ==="

# Load Python directly from ACCRE modules
module purge || true
module load Python/3.10.8-GCCcore-12.2.0 || module load Python/3.10 || module load python/3.10 || true

# Check Python availability
if command -v python3 >/dev/null 2>&1; then
    echo "✓ Python found: $(python3 --version)"
elif command -v python >/dev/null 2>&1; then
    echo "✓ Python found: $(python --version)"
else
    echo "ERROR: No Python found"
    exit 1
fi

# Check required files
MODEL="ucf11_c3d_16f_onnex.onnx"
SCRIPT="verify_ucf11_c3d.py"

echo "Checking for required files..."
[[ -f "${MODEL}" ]]     || { echo "ERROR: Missing ${MODEL}"; exit 2; }
[[ -f "${SCRIPT}" ]]    || { echo "ERROR: Missing ${SCRIPT}"; exit 3; }

echo "✓ Model: ${MODEL} ($(du -h ${MODEL} | cut -f1))"
echo "✓ Script: ${SCRIPT}"

# Set up Python environment
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Try to install required packages if possible
echo "Installing required packages..."
pip3 install --user numpy scipy matplotlib onnx onnxruntime 2>/dev/null || true

echo "Starting native verification at $(date)..."

# Run verification directly with Python
set +e
python3 "${SCRIPT}"
EC=$?
set -e

echo "Native verification completed at $(date) with exit code: ${EC}"

if [[ ${EC} -eq 0 ]]; then
    echo "✓ SUCCESS: Verification completed successfully"
else
    echo "✗ FAILED: Verification failed with exit code ${EC}"
fi

exit ${EC}
