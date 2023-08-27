#!/bin/bash
#SBATCH --chdir /home/bigi/LE-ACE-generalized-cgs/
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 72
#SBATCH --partition bigmem
#SBATCH --mem 980G
#SBATCH --time 3-0

echo STARTING AT `date`
python -u linear_fit.py $1 $2 $3 $4 $5 $6 $7
echo FINISHED at `date`
