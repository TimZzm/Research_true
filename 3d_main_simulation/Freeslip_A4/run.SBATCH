#!/bin/bash

#SBATCH --nodes=2
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --time=6:00:00
#SBATCH --job-name='3d_A4_Rstd'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zz3645@nyu.edu
#SBATCH --output='3d_A4_Rstd.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 'A4_Rstd.py' 