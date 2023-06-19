#!/usr/bin/bash

#SBATCH --partition=elec.gpu.q
#SBATCH --output=train_log3.out

module load cuda10.2/toolkit/10.2.89

echo "Process with PID 11682234 has ended. Executing command now."
