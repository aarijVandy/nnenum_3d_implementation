#!/bin/bash

# Script to build nnenum_verification.sif off-cluster
# Run this on your local machine or a system with Docker/Apptainer access

set -euo pipefail

SIF_IMAGE="nnenum_verification.sif"
DEF_FILE="nnenum.def"

echo "Building Apptainer SIF image off-cluster..."
echo "This should be run on a system with Apptainer/Singularity installed"

# Check if apptainer/singularity is available
if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_CMD="apptainer"
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_CMD="singularity"
else
    echo "ERROR: Neither apptainer nor singularity found"
    echo "Please install Apptainer/Singularity first"
    exit 1
fi

echo "Using: $CONTAINER_CMD"

# Check if definition file exists
if [[ ! -f "$DEF_FILE" ]]; then
    echo "ERROR: Definition file $DEF_FILE not found"
    exit 2
fi

# Build the SIF image
echo "Building $SIF_IMAGE from $DEF_FILE..."
$CONTAINER_CMD build --force "$SIF_IMAGE" "$DEF_FILE"

# Check if build was successful
if [[ -f "$SIF_IMAGE" ]]; then
    echo "SUCCESS: $SIF_IMAGE built successfully"
    echo "Size: $(du -h "$SIF_IMAGE" | cut -f1)"
    echo ""
    echo "Next steps:"
    echo "1. Transfer $SIF_IMAGE to the ACCRE cluster"
    echo "2. Place it in your nnenum working directory"
    echo "3. Run your SLURM job"
    echo ""
    echo "Transfer command example:"
    echo "scp $SIF_IMAGE username@login.accre.vanderbilt.edu:~/nnenum/"
else
    echo "ERROR: Build failed - $SIF_IMAGE not created"
    exit 3
fi
