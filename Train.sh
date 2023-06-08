#!/usr/bin/bash

#SBATCH --partition=elec.gpu.q
#SBATCH --output=First_CPC_w_AR.out
#SBATCH --gres gpu:1
module load cuda10.2/toolkit/10.2.89

echo "Calling python file"

python CPC_train.py 
