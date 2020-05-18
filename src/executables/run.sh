#!/bin/bash

#SBATCH -N 1
#SBATCH -p gpu_small
#SBATCH -J NODESolver
#SBATCH --output=out.txt
#SBATCH -e err.txt

#SBATCH --gres=gpu:1
#SBATCH --time=7:00:00

module load python/pytorch-1.5.0

echo '>>> running the script >>>'
srun python3 main.py 
echo '<<< running the script <<<'
