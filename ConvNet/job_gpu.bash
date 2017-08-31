#!/bin/bash 

#SBATCH -N 1 # node count 
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH -t 2:00:00 
#SBATCH --gres=gpu:1

echo "pwd: `pwd`"
echo "Start time: `date`"

python nn_cuda.py acc

echo "End time: `date`"
