#!/bin/bash
#SBATCH --job-name=ucf11_c3d_verification
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Required files
SIF_IMAGE="nnenum_verification.sif"
MODEL_PATH="ucf11_c3d_16f_onnex.onnx"
RUN_LOG="verification_output_${SLURM_JOB_ID:-manual}.log"

# Apptainer local cache/tmp only in current dir
export APPTAINER_CACHEDIR="$(pwd)/.apptainer_cache_${SLURM_JOB_ID:-run}"
export APPTAINER_TMPDIR="$(pwd)/.apptainer_tmp_${SLURM_JOB_ID:-run}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# Ensure apptainer
if ! command -v apptainer >/dev/null 2>&1; then
  module load Apptainer 2>/dev/null || module load apptainer 2>/dev/null || true
fi
command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not found"; exit 127; }

# Checks
[ -f "${SIF_IMAGE}" ]  || { echo "ERROR: missing image: ${SIF_IMAGE}"; exit 2; }
[ -f "${MODEL_PATH}" ] || { echo "ERROR: missing model: ${MODEL_PATH}"; exit 3; }

# Threading env (optional)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "=== START $(date) ==="    | tee "${RUN_LOG}"
echo "Host=$(hostname) CWD=$(pwd)" | tee -a "${RUN_LOG}"
echo "SIF=${SIF_IMAGE} MODEL=${MODEL_PATH}" | tee -a "${RUN_LOG}"

# Portable timing wrapper (no GNU time, no -v needed)
CMD='START=$(date +%s);
python verify_ucf11_c3d.py;
EC=$?;
END=$(date +%s);
echo "WALLCLOCK=$((END-START))s";
exit $EC'

set +e
apptainer exec --cleanenv --bind "$(pwd):/work" --pwd /work \
  "${SIF_IMAGE}" bash -lc "${CMD}" >> "${RUN_LOG}
