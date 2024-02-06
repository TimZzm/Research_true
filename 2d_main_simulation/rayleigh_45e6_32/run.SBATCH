#!/bin/bash

#SBATCH --nodes=2
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
#SBATCH --job-name='wdrmrbc_ra_45e6'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zz3645@nyu.edu
#SBATCH --output='wdrmrbc_ra_45e6.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 'wdrmrbc_ra_45e6.py' 