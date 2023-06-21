#!/usr/bin/bash

#SBATCH --partition=elec.gpu.q
#SBATCH --output=train_CPC_log_4.out
#SBATCH --gres gpu:1
module load cuda10.2/toolkit/10.2.89

echo "Calling python file"
source activate 5LSM0
# For boolean flags, do not include in the command line. 
# Just include the flag to make the variable evaluate to True. It is false by default

for i in {1..10}; 
do
    CHECKPOINT_FILE="/home/tue/20191313/5aua0-2022-group-18/trained_models/CPC_candidates/CPC_random_search_2_1/ckpt_20epochs.pth"
    python Train.py  --epochs 20 --random_search --freeze_encoder --output_name "1_CPC_2_1_random_search_$i"\
        --load_checkpoint $CHECKPOINT_FILE
done
