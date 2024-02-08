#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40GB
#SBATCH --time=6:00:00
#SBATCH --job-name='vid'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zz3645@nyu.edu
#SBATCH --output='vid.out'

module purge

srun /scratch/work/public/singularity/run-dedalus-3.0.0a0.bash python 'video.py' 