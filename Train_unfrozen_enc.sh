#!/usr/bin/bash

#SBATCH --partition=elec.gpu.q
#SBATCH --output=train_unfrozen_enc_log.out
#SBATCH --gres gpu:1
module load cuda10.2/toolkit/10.2.89

echo "Calling python file"
source activate 5LSM0
# For boolean flags, do not include in the command line. 
# Just include the flag to make the variable evaluate to True. It is false by default



for {1..30}; 
do
    CHECKPOINT_FILE="/home/tue/20191313/5aua0-2022-group-18/trained_models/CPC_random_search_$i/ckpt_20epochs.pth"
    python Train.py  --batch_size_train 64 --epochs 10 \
                     --lr 0.0005 --max_lr 0.002 --output_name "Classifier_CPC_frozen_from_$i"\
                     --load_checkpoint $CHECKPOINT_FILE
done

