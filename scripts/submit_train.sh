# !/bin/bash

NUM_RUNS=5

for RUN in `seq $NUM_RUNS`; do 
   sbatch -D ./slurm-logs/ --job-name=cd_$RUN \
       --export="PYFILE=src/vit_train.py,ARGS=cadmium_24x_t120_baseline" \
       scripts/gpu_job.sh
done
