#!/usr/bin/bash

#SBATCH --partition=elec.gpu.q
#SBATCH --output=train_log.out
#SBATCH --gres gpu:1
module load cuda10.2/toolkit/10.2.89

echo "Calling python file"
source activate 5LSM0
# For boolean flags, do not include in the command line. 
# Just include the flag to make the variable evaluate to True. It is false by default

for i in {1..10}; 
do
    python Train.py  --replicate_CPC_params --epochs 20 --random_search --output_name "FS_random_search_third_run_$i"
done
