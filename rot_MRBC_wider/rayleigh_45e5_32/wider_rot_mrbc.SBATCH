#!/bin/bash

#SBATCH --nodes=10
#SBATCH --tasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=2:00:00
#SBATCH --job-name='wdrmrbc'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zz3645@nyu.edu
#SBATCH --output='wdrmrbc.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 'wdrmrbc.py' 