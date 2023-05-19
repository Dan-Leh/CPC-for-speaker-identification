#!/usr/bin/bash

#SBATCH --partition=elec.gpu.q
#SBATCH --output=first_run.out
#SBATCH --gres gpu:1
module load cuda10.2/toolkit/10.2.89

echo "Calling python file"

python FS_train.py 
