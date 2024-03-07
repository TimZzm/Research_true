#!/bin/bash

#SBATCH --nodes=3
#SBATCH --tasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=288GB
#SBATCH --time=50:00:00
#SBATCH --job-name='3d_A9_R_p2'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zz3645@nyu.edu
#SBATCH --output='3d_A9_R_p2.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 'A9_R_p2.py' 