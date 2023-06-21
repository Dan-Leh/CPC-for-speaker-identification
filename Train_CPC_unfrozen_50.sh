#!/usr/bin/bash

#SBATCH --partition=elec.gpu.q
#SBATCH --output=train_CPC_unfrozen_50.out
#SBATCH --gres gpu:1
module load cuda10.2/toolkit/10.2.89

echo "Calling python file"
source activate 5LSM0
# For boolean flags, do not include in the command line. 
# Just include the flag to make the variable evaluate to True. It is false by default
CHECKPOINT_FILE="/home/tue/20191313/5aua0-2022-group-18/trained_models/CPC_candidates/CPC_enc_100epoch/ckpt_100epochs.pth"
python Train.py --epochs 100 --lr 0.00310303348696595 --max_lr 0.028885041334915096 --batch_size 61 --output_name "CPC_unfrozen_50"\
    --data_percentage 50 --n_predictions 5 --n_negatives 16 --n_past_latents 1 --replicate_CPC_params --checkpoint_file $CHECKPOINT_FILE