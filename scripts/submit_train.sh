# !/bin/bash

NUM_RUNS=5

# New data set with new training model
for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=24x_merged_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=24x_120_merged" \
        scripts/gpu_job.sh
done
