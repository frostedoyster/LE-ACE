#!/bin/bash
#SBATCH --chdir /home/bigi/LE-ACE/
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 28
#SBATCH --mem 245G
#SBATCH --time 2-0 
export OMP_NUM_THREADS=28
echo STARTING AT `date`
python -u main.py inputs/methane1.json > outputs/methane1b.out
echo FINISHED at `date`