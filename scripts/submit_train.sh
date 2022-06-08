# !/bin/bash

NUM_RUNS=2

for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=t50_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=cell_cycle_24x_v100_t50_b96_baseline" \
        scripts/gpu_job.sh
done
