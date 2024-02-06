#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --time=3:00:00
#SBATCH --job-name='wdrmrbc_ra_1e7'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zz3645@nyu.edu
#SBATCH --output='wdrmrbc_ra_1e7.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 'wdrmrbc_ra_1e7.py' 