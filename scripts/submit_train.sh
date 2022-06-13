# !/bin/bash

NUM_RUNS=2

for RUN in `seq $NUM_RUNS`; do 
   sbatch -D ./slurm-logs/ --job-name=cdh_$RUN \
       --export="PYFILE=src/vit_train.py,ARGS=cadmium_24x_hybrid_simple" \
       scripts/gpu_job.sh
done

for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=cch_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=cell_cycle_24x_hybrid_simple" \
        scripts/gpu_job.sh
done
