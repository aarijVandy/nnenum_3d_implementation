#!/bin/bash
#SBATCH --job-name=ucf11_verify_final
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=verify-%j.out
#SBATCH --error=verify-%j.err

# CRITICAL: Set ACCRE variables BEFORE strict mode to avoid "unbound variable" errors
export USER_IS_ROOT="${USER_IS_ROOT:-0}"

# Initialize ACCRE software stack early (before strict mode)
if type setup_accre_software_stack &>/dev/null; then
    setup_accre_software_stack || true
fi

# NOW enable strict mode
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

echo "=== UCF11 VERIFICATION $(date) ==="

# Load Apptainer
module purge || true
module load Apptainer 2>/dev/null || module load apptainer 2>/dev/null || true
command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not found"; exit 127; }

# Check required files
SIF_IMAGE="nnenum_verification.sif"
MODEL="ucf11_c3d_16f_onnex.onnx"
SCRIPT="verify_ucf11_c3d.py"

echo "Checking for required files..."
[[ -f "${SIF_IMAGE}" ]] || { echo "ERROR: Missing ${SIF_IMAGE}"; ls -la *.sif; exit 2; }
[[ -f "${MODEL}" ]]     || { echo "ERROR: Missing ${MODEL}"; exit 3; }
[[ -f "${SCRIPT}" ]]    || { echo "ERROR: Missing ${SCRIPT}"; exit 4; }

echo "✓ Container: ${SIF_IMAGE} ($(du -h ${SIF_IMAGE} | cut -f1))"
echo "✓ Model: ${MODEL} ($(du -h ${MODEL} | cut -f1))"
echo "✓ Script: ${SCRIPT}"

# Set up environment
export APPTAINER_CACHEDIR="$(pwd)/.cache_verify"
export APPTAINER_TMPDIR="$(pwd)/.tmp_verify"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Starting verification at $(date)..."

# Run verification
set +e
apptainer exec --cleanenv --bind "$(pwd):/work" --pwd /work \
  "${SIF_IMAGE}" python "${SCRIPT}"
EC=$?
set -e

echo "Verification completed at $(date) with exit code: ${EC}"

# Cleanup
rm -rf "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" 2>/dev/null || true

if [[ ${EC} -eq 0 ]]; then
    echo "✓ SUCCESS: Verification completed successfully"
else
    echo "✗ FAILED: Verification failed with exit code ${EC}"
fi

exit ${EC}
