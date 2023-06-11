#!/usr/bin/bash

#SBATCH --partition=elec.gpu.q
#SBATCH --output=train_log.out
#SBATCH --gres gpu:1
module load cuda10.2/toolkit/10.2.89

echo "Calling python file"

# For boolean flags, do not include in the command line. 
# Just include the flag to make the variable evaluate to True. It is false by default

python Train.py  --epoch 2 --lr 0.01
python Train.py  --CPC --epoch 2
