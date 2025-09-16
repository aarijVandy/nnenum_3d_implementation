#!/bin/bash
#SBATCH --job-name=ucf11_verify
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.err

# Start strict, but NOT nounset yet (avoids Lmod clash)
set -Eeo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ---- Your filenames (drop-in) ----
SIF_IMAGE="${SIF_IMAGE:-nnenum_verification.sif}"
SCRIPT="${SCRIPT:-verify_ucf11_c3d.py}"
MODEL="${MODEL:-ucf11_c3d_16f_onnex.onnx}"

RUN_ID="${SLURM_JOB_ID:-manual}"
RUN_LOG="verification_output_${RUN_ID}.log"

# ---- Make Lmod happy under -u later ----
# (safe even if USER_IS_ROOT was unset)
export USER_IS_ROOT="${USER_IS_ROOT:-0}"

# ---- Initialize ACCRE stack (quiet if missing) ----
type setup_accre_software_stack &>/dev/null && setup_accre_software_stack || true

# ---- Load Apptainer; guard in case 'module' not available ----
if command -v module &>/dev/null; then
  # Lmod init scripts don't expect nounset; keep -u off here
  module purge || true
  module load apptainer 2>/dev/null || module load Apptainer 2>/dev/null || true
fi

# Fallback: absolute path if modules didn’t add it
if ! command -v apptainer >/dev/null 2>&1; then
  if [[ -x /usr/bin/apptainer ]]; then
    PATH="/usr/bin:$PATH"
  fi
fi

command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not found"; exit 127; }

# ---- Now go fully strict (nounset on) ----
set -u

# ---- Sanity checks ----
[[ -f "${SIF_IMAGE}" ]] || { echo "ERROR: Missing ${SIF_IMAGE}"; exit 2; }
[[ -f "${SCRIPT}"    ]] || { echo "ERROR: Missing ${SCRIPT}"; exit 3; }
[[ -f "${MODEL}"     ]] || { echo "ERROR: Missing ${MODEL}"; exit 4; }

# ---- Apptainer cache/tmp in CWD (avoid /tmp quotas) ----
APPT_CACHE="${PWD}/.appt_cache_${RUN_ID}"
APPT_TMP="${PWD}/.appt_tmp_${RUN_ID}"
mkdir -p "${APPT_CACHE}" "${APPT_TMP}" .cache
export APPTAINER_CACHEDIR="${APPT_CACHE}"
export APPTAINER_TMPDIR="${APPT_TMP}"

# Threads into container
CPUS="${SLURM_CPUS_PER_TASK:-1}"
export APPTAINERENV_OMP_NUM_THREADS="${CPUS}"
export APPTAINERENV_MKL_NUM_THREADS="${CPUS}"
export APPTAINERENV_NUMBA_NUM_THREADS="${CPUS}"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_XDG_CACHE_HOME="/work/.cache"

umask 077

# ---- Footer/cleanup trap ----
on_exit() {
  local ec=$?
  echo "=== END $(date) EXIT=${ec} ===" | tee -a "${RUN_LOG}"
  rm -rf "${APPT_CACHE}" "${APPT_TMP}" || true
  exit "${ec}"
}
trap on_exit EXIT

# ---- Header ----
{
  echo "=== START $(date) on $(hostname) ==="
  echo "CWD: ${PWD}"
  echo "Apptainer: $(apptainer --version)"
  echo "SLURM_CPUS_PER_TASK=${CPUS}"
  echo "Image: ${SIF_IMAGE}"
  echo "Script: ${SCRIPT}"
  echo "Model: ${MODEL}"
} | tee "${RUN_LOG}"

# ---- Run once ----
set +e
srun -n 1 apptainer exec --cleanenv \
  --bind "${PWD}:/work" --pwd /work \
  "${SIF_IMAGE}" python3 -u "${SCRIPT}" --model "${MODEL}" >> "${RUN_LOG}" 2>&1
EC=$?
set -e

exit "${EC}"
