# !/bin/bash

NUM_RUNS=5

# Cell cycle run
for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=24x_comp_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=cell_cycle_24x_chr4" \
        scripts/gpu_job.sh
done
