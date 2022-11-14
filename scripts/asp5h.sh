#!/bin/bash
#SBATCH --chdir /home/bigi/LE-ACE-clever/
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 180G
#SBATCH --time 2-0 
export OMP_NUM_THREADS=32
echo STARTING AT `date`
python -u main.py inputs/asp5.json > outputs/asp5h.out
echo FINISHED at `date`