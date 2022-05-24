# !/bin/bash

NUM_RUNS=1

for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=v_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=simple" \
        scripts/gpu_job.sh
done
