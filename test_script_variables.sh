#!/bin/bash

# Test script to verify the unbound variable fix
set -u  # Same strict mode as the main script

# Initialize variables as in the fixed script
APPTAINER_CACHEDIR=""
APPTAINER_TMPDIR=""

echo "Testing variable initialization..."

# Test the cleanup section logic
if [ -n "${APPTAINER_CACHEDIR}" ] && [ -d "${APPTAINER_CACHEDIR}" ]; then
  echo "Would clean up cache directory: ${APPTAINER_CACHEDIR}"
else
  echo "No cache directory to clean up (this is expected)"
fi

if [ -n "${APPTAINER_TMPDIR}" ] && [ -d "${APPTAINER_TMPDIR}" ]; then
  echo "Would clean up temp directory: ${APPTAINER_TMPDIR}"  
else
  echo "No temp directory to clean up (this is expected)"
fi

echo "Test completed successfully - no unbound variable errors!"
