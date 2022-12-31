#!/bin/bash
#
#SBATCH --job-name=jet_5km
#SBATCH --output=output_%j.txt
#SBATCH --ntasks-per-node=28
##SBATCH --nodes=12
##SBATCH --time=3:00:00
##SBATCH -p short-28core
#SBATCH --nodes=12
#SBATCH --time=12:00:00
#SBATCH -p medium-28core
##SBATCH --nodes=8
##SBATCH --time=48:00:00
##SBATCH -p long-28core
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=christopher.wolfe@stonybrook.edu

# This configuration will do a year in about 1.4 hours on 8 nodes (224 tasks) writing daily diagnostics
# This configuration will do a year in about 0.95 hours on 12 nodes (336 tasks) writing daily diagnostics (nearly linear speedup)

# short-28core: max length 4 hours, max nodes 12
# medium-28core: max length 12 hours, max nodes 24
# long-28core: max length 48 hours, max nodes 8


module load intel/compiler/64/2020/20.0.2
module load intel/mkl/64/2020/20.0.2
module load intel/mpi/64/2020/20.0.2
module load netcdf-fortran

cd $SLURM_SUBMIT_DIR
mkdir -p OUTPUT
mkdir -p RESTART

date
mpirun ./MOM6
date
