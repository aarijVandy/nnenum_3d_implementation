#!/bin/bash
#SBATCH --job-name=ucf11_verification_CLEAN
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.err

# CRITICAL: Set ACCRE variables BEFORE strict mode to avoid "unbound variable" errors
export USER_IS_ROOT="${USER_IS_ROOT:-0}"

# Initialize ACCRE software stack early (before strict mode)  
if type setup_accre_software_stack &>/dev/null; then
    setup_accre_software_stack || true
fi

# NOW enable strict mode
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Files needed
SIF_IMAGE="nnenum_verification.sif"
MODEL="ucf11_c3d_16f_onnex.onnx"
RUN_LOG="verification_output_${SLURM_JOB_ID:-manual}.log"

# Load Apptainer
module purge || true
module load Apptainer 2>/dev/null || module load apptainer 2>/dev/null || true
command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not found"; exit 127; }

# Check files exist
[[ -f "${SIF_IMAGE}" ]] || { echo "ERROR: Missing ${SIF_IMAGE}"; exit 2; }
[[ -f "${MODEL}" ]]     || { echo "ERROR: Missing ${MODEL}"; exit 3; }

# Local cache dirs to avoid /tmp issues
export APPTAINER_CACHEDIR="$(pwd)/.cache_${SLURM_JOB_ID:-run}"
export APPTAINER_TMPDIR="$(pwd)/.tmp_${SLURM_JOB_ID:-run}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# Threading control
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Start log
echo "=== START $(date) ===" | tee "${RUN_LOG}"
echo "Container: ${SIF_IMAGE}" | tee -a "${RUN_LOG}"
echo "Model: ${MODEL}" | tee -a "${RUN_LOG}"
echo "Running verification..." | tee -a "${RUN_LOG}"

# Run the verification - use variables, not hardcoded names
set +e
apptainer exec --cleanenv --bind "$(pwd):/work" --pwd /work \
  "${SIF_IMAGE}" python verify_ucf11_c3d.py >> "${RUN_LOG}" 2>&1
EC=$?
set -e

echo "=== END $(date) EXIT=${EC} ===" | tee -a "${RUN_LOG}"

# Cleanup
rm -rf "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" 2>/dev/null || true

exit "${EC}"
