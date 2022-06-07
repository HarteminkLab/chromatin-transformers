# !/bin/bash

NUM_RUNS=2

for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=cc_simple_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=cell_cycle_24x_chr4_validchrom_b96_simple_p20" \
        scripts/gpu_job.sh
done
