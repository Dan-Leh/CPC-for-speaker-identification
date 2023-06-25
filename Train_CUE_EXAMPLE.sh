#!/usr/bin/bash

#SBATCH --partition=elec.gpu.q
#SBATCH --output=CUE_EXAMPLE_log.out
#SBATCH --gres gpu:1
module load cuda10.2/toolkit/10.2.89

echo "Calling python file"
source activate 5LSM0

# Specify your path to the checkpoit file:  ".../trained_models/P-CPC_EXAMPLE/ckpt_XXepochs.pth"
CHECKPOINT_FILE= "<insert path here>"

# For boolean flags, do not include in the command line. 
# Just include the flag to make the variable evaluate to True. It is false by default
python Train.py --epochs 10 --lr 0.00310303348696595 --max_lr 0.028885041334915096 --batch_size 61 --output_name "CUE_EXAMPLE"\
    --data_percentage 100 --n_predictions 5 --n_negatives 16 --n_past_latents 1 --replicate_CPC_params --load_checkpoint $CHECKPOINT_FILE