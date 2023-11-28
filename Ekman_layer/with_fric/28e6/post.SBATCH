#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=01:00:00
#SBATCH --job-name='post_with_fric_28e6'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zz3645@nyu.edu
#SBATCH --output='post_with_fric_28e6.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 'postanalysis.py' 