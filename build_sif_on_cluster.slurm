#!/bin/bash
#SBATCH --job-name=build_sif
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=build-sif-%j.out
#SBATCH --error=build-sif-%j.err

# CRITICAL: Set ACCRE variables BEFORE strict mode to avoid "unbound variable" errors
export USER_IS_ROOT="${USER_IS_ROOT:-0}"

# Initialize ACCRE software stack early (before strict mode)
if type setup_accre_software_stack &>/dev/null; then
    setup_accre_software_stack || true
fi

# NOW enable strict mode
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

echo "=== BUILDING SIF CONTAINER $(date) ==="

# Load Apptainer
module purge || true
module load Apptainer 2>/dev/null || module load apptainer 2>/dev/null || true
command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not found"; exit 127; }

echo "Apptainer version: $(apptainer --version)"

# Check if definition file exists
[[ -f "nnenum.def" ]] || { echo "ERROR: Missing nnenum.def file"; exit 2; }

# Use local node storage instead of PANFS to avoid permission issues
# CRITICAL: Use /tmp on the local node, not the shared PANFS filesystem
export APPTAINER_CACHEDIR="/tmp/apptainer_cache_${SLURM_JOB_ID:-$$}"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp_${SLURM_JOB_ID:-$$}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# Set proper permissions for the cache directories
chmod 755 "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

echo "Building nnenum_verification.sif from nnenum.def..."

# Build the container with additional flags for permission issues
echo "Building nnenum_verification.sif from nnenum.def..."

# Use --fakeroot to avoid permission issues, and --disable-cache as fallback
apptainer build --fakeroot nnenum_verification.sif nnenum.def

echo "Build completed. Container info:"
ls -lh nnenum_verification.sif

# Test the container quickly
echo "Testing container..."
apptainer exec nnenum_verification.sif python --version

echo "=== BUILD COMPLETE $(date) ==="

# Cleanup
rm -rf "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" 2>/dev/null || true
