# !/bin/bash

NUM_RUNS=2

for RUN in `seq $NUM_RUNS`; do 
   sbatch -D ./slurm-logs/ --job-name=sim_$RUN \
       --export="PYFILE=src/vit_train.py,ARGS=cadmium_24x_random_simple_d2" \
       scripts/gpu_job.sh
done

for RUN in `seq $NUM_RUNS`; do 
   sbatch -D ./slurm-logs/ --job-name=sim_$RUN \
       --export="PYFILE=src/vit_train.py,ARGS=cadmium_24x_random_simple_d4" \
       scripts/gpu_job.sh
done

for RUN in `seq $NUM_RUNS`; do 
   sbatch -D ./slurm-logs/ --job-name=sim_$RUN \
       --export="PYFILE=src/vit_train.py,ARGS=cadmium_24x_random_simple_d8" \
       scripts/gpu_job.sh
done

for RUN in `seq $NUM_RUNS`; do 
   sbatch -D ./slurm-logs/ --job-name=sim_$RUN \
       --export="PYFILE=src/vit_train.py,ARGS=cadmium_24x_random_simple_d16" \
       scripts/gpu_job.sh
done
