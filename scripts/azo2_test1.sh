#!/bin/bash
#SBATCH --chdir /home/bigi/LE-ACE-clever/
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 180G
#SBATCH --time 2-0 
export OMP_NUM_THREADS=32
echo STARTING AT `date`
python -u main.py inputs/azo2_test1.json > outputs/azo2_test1.out
echo FINISHED at `date`