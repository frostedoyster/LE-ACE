#!/bin/bash
#SBATCH --chdir /home/bigi/LE-ACE-generalized-cgs/
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --gres gpu:1
#SBATCH --mem 180G
#SBATCH --time 1:00:00 

username="bigi"
le_ace_directory="LE-ACE-generalized-cgs/"
datetime=$(date +%Y-%m-%d-%k-%M-%S-%N)

echo STARTING AT `date`

python -u calculate_features.py $1 $2 $3 $datetime

cd "/scratch/izar/$username/"

for file in ./*; do
    if [[ -f "$file" ]]; then
        if [[ $file =~ $datetime ]]; then
            IFS='-' read -ra split_string <<< "$file"
            molecule="${split_string[1]}"
        fi
    fi
done

sftp $username@jed.epfl.ch << EOF 
cd /scratch/$username/ 
put X_train-$molecule-$datetime.pt 
put X_test-$molecule-$datetime.pt
put y_train-$molecule-$datetime.pt 
put y_test-$molecule-$datetime.pt
put LE_reg-$molecule-$datetime.pt
EOF

ssh $username@jed.epfl.ch << EOF
conda deactivate
module load gcc python
source virtualenv-j/bin/activate
cd /home/$username/$le_ace_directory/
sbatch linear_fit.sh /scratch/$username/X_train-$molecule-$datetime.pt /scratch/$username/X_test-$molecule-$datetime.pt /scratch/$username/y_train-$molecule-$datetime.pt /scratch/$username/y_test-$molecule-$datetime.pt /scratch/$username/LE_reg-$molecule-$datetime.pt $molecule $4
EOF

echo FINISHED at `date`
