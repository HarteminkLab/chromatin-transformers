# !/bin/bash

NUM_RUNS=5

for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=c_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=simple" \
        scripts/gpu_job.sh
done

for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=c_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=complex" \
        scripts/gpu_job.sh
done

for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=c_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=complex_120" \
        scripts/gpu_job.sh
done

