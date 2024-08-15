#!/bin/bash
#SBATCH --partition h100
#SBATCH --exclude=kl002,kl006,kl014
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --gres gpu:1
#SBATCH --mem 80G
#SBATCH --time 1-0

source /scratch/bigi/virtualenv-k/bin/activate

echo STARTING AT `date`
echo $1
python -u carbon.py
echo FINISHED at `date`
