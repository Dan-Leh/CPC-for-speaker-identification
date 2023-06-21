#!/usr/bin/bash

#SBATCH --partition=elec.gpu.q
#SBATCH --output=train_FS_1_log.out
#SBATCH --gres gpu:1
module load cuda10.2/toolkit/10.2.89

echo "Calling python file"
source activate 5LSM0
# For boolean flags, do not include in the command line. 
# Just include the flag to make the variable evaluate to True. It is false by default


python Train.py --data_percentage 1 --epochs 100 --n_predictions 5 --n_negatives 16 --n_past_latents 1 --replicate_CPC_params --batch_size 26 --lr 0.0036679001150894475 --max_lr 0.005722272739232646 --output_name "Fully_supervised_1"

