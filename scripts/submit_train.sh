# !/bin/bash

NUM_RUNS=5

for RUN in `seq $NUM_RUNS`; do 
   sbatch -D ./slurm-logs/ --job-name=lfc_$RUN \
       --export="PYFILE=src/vit_train.py,ARGS=cd_24x128_2chan_p1_logfold" \
       scripts/gpu_job.sh
done
