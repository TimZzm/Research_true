#!/bin/bash

#SBATCH --nodes=8
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --job-name='with_fric_28e6'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zz3645@nyu.edu
#SBATCH --output='with_fric_28e6.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 'test_with_fric.py' 