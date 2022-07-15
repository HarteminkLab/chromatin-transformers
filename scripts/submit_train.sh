# !/bin/bash

NUM_RUNS=3

for RUN in `seq $NUM_RUNS`; do 
   sbatch -D ./slurm-logs/ --job-name=lr_$RUN \
       --export="PYFILE=src/vit_train.py,ARGS=cd_24x_long_running" \
       scripts/gpu_job.sh
done
