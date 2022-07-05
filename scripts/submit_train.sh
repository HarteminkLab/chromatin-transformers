# !/bin/bash

NUM_RUNS=5

for RUN in `seq $NUM_RUNS`; do 
   sbatch -D ./slurm-logs/ --job-name=c1_$RUN \
       --export="PYFILE=src/vit_train.py,ARGS=cd_24x_complex" \
       scripts/gpu_job.sh
done

for RUN in `seq $NUM_RUNS`; do 
   sbatch -D ./slurm-logs/ --job-name=c2_$RUN \
       --export="PYFILE=src/vit_train.py,ARGS=cd_24x_more_complex" \
       scripts/gpu_job.sh
done

for RUN in `seq $NUM_RUNS`; do 
   sbatch -D ./slurm-logs/ --job-name=c3_$RUN \
       --export="PYFILE=src/vit_train.py,ARGS=cd_24x_even_more_complex" \
       scripts/gpu_job.sh
done
