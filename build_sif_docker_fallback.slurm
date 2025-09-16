#!/bin/bash
#SBATCH --job-name=build_sif_docker
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=build-docker-%j.out
#SBATCH --error=build-docker-%j.err

# CRITICAL: Set ACCRE variables BEFORE strict mode
export USER_IS_ROOT="${USER_IS_ROOT:-0}"

# Initialize ACCRE software stack early
if type setup_accre_software_stack &>/dev/null; then
    setup_accre_software_stack || true
fi

# NOW enable strict mode
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

echo "=== BUILDING SIF FROM DOCKER HUB $(date) ==="

# Load Apptainer
module purge || true
module load Apptainer 2>/dev/null || module load apptainer 2>/dev/null || true
command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not found"; exit 127; }

echo "Apptainer version: $(apptainer --version)"

# Use local node storage to avoid PANFS permission issues
export APPTAINER_CACHEDIR="/tmp/apptainer_cache_${SLURM_JOB_ID:-$$}"
export APPTAINER_TMPDIR="/tmp/apptainer_tmp_${SLURM_JOB_ID:-$$}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
chmod 755 "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# Try pulling from a standard Python image and building manually
echo "Building from Docker Hub python:3.10 image..."

# Build directly from Docker Hub
apptainer build --fakeroot nnenum_verification.sif docker://python:3.10

echo "Container built successfully. Now customizing..."

# Create a simple overlay script to install dependencies
cat > customize_container.sh << 'EOF'
#!/bin/bash
cd /work
pip3 install -r requirements.txt || pip3 install numpy scipy matplotlib onnx onnxruntime
export PYTHONPATH=$PYTHONPATH:/work/src
EOF

chmod +x customize_container.sh

echo "Testing basic container functionality..."
apptainer exec nnenum_verification.sif python --version

echo "Container info:"
ls -lh nnenum_verification.sif

echo "=== BUILD COMPLETE $(date) ==="

# Cleanup
rm -rf "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" 2>/dev/null || true
rm -f customize_container.sh
