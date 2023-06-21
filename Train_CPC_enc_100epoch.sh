#!/usr/bin/bash

#SBATCH --partition=elec.gpu.q
#SBATCH --output=train_CPC_100epochs.out
#SBATCH --gres gpu:1
module load cuda10.2/toolkit/10.2.89

echo "Calling python file"
source activate 5LSM0
# For boolean flags, do not include in the command line. 
# Just include the flag to make the variable evaluate to True. It is false by default

python Train.py --epochs 100 --CPC --lr 0.004927021583315313 --max_lr 0.03261372807491948 --batch_size 33 --output_name "CPC_enc_100epoch"\
    --n_predictions 5 --n_negatives 16 --n_past_latents 1