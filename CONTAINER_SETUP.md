# ACCRE Container Setup Instructions

## Current Status

Based on your error logs, you have `nnenum_verification.sif` (812M) already built on the cluster. However, according to ACCRE guidelines, container images should be built **off-cluster** for security and best practices.

## Option 1: Use Existing SIF File (Quick)

If the existing `nnenum_verification.sif` on the cluster works:
1. Simply run your updated SLURM script: `sbatch run_ucf11_verification.slurm`
2. The script now follows ACCRE guidelines properly

## Option 2: Build SIF Off-Cluster (Recommended)

For best practices, build the SIF file off-cluster:

### Prerequisites
- Local machine or server with Apptainer/Singularity installed
- Access to transfer files to ACCRE

### Steps
1. **On your local machine:**
   ```bash
   # Build the SIF file
   ./build_sif_offcluster.sh
   ```

2. **Transfer to ACCRE:**
   ```bash
   scp nnenum_verification.sif username@login.accre.vanderbilt.edu:~/nnenum/
   ```

3. **On ACCRE, submit job:**
   ```bash
   sbatch run_ucf11_verification.slurm
   ```

## Key Changes Made

The updated `run_ucf11_verification.slurm` script now:
- ✅ Uses `apptainer exec` instead of shell commands
- ✅ Properly binds directories with `--bind`
- ✅ Removes unsupported `-v` flag from time command
- ✅ Uses local cache/tmp directories
- ✅ Follows ACCRE's recommended patterns
- ✅ Simplified output (no directory listings)
- ✅ Proper cleanup of temporary directories

## Files Updated

1. `run_ucf11_verification.slurm` - Streamlined SLURM script following ACCRE guidelines
2. `build_sif_offcluster.sh` - Script to build SIF file off-cluster
3. `CONTAINER_SETUP.md` - This instruction file

## Expected Output

With these changes, you should see:
- Exit code 0 on successful verification
- Clean log output in `verification_output_<job_id>.log`
- Proper error messages if issues occur
