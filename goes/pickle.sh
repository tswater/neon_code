#!/bin/sh
#SBATCH --nodes=3
#SBATCH --tasks-per-node=2
#SBATCH --job-name="goes_neon"
#SBATCH --output="log.txt"
#SBATCH --exclusive
#SBATCH --exclude=node5,node6,node7,node8
time mpiexec -n 96 python create_pickles.py

