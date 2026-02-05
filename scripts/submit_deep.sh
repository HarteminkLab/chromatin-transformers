# !/bin/bash

NUM_RUNS=2

for RUN in `seq $NUM_RUNS`; do 
   sbatch -D ./slurm-logs/ --job-name=dp_$RUN \
       --export="PYFILE=src/vit_cluster_train.py,ARGS=deep_clustering" \
       scripts/gpu_job.sh
done

