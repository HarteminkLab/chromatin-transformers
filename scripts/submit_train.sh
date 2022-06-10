# !/bin/bash

NUM_RUNS=2

# for RUN in `seq $NUM_RUNS`; do 
#     sbatch -D ./slurm-logs/ --job-name=t50_$RUN \
#         --export="PYFILE=src/vit_train.py,ARGS=cell_cycle_24x_random_baseline" \
#         scripts/gpu_job.sh
# done

# for RUN in `seq $NUM_RUNS`; do 
#     sbatch -D ./slurm-logs/ --job-name=t50_$RUN \
#         --export="PYFILE=src/vit_train.py,ARGS=cell_cycle_24x_random_simple" \
#         scripts/gpu_job.sh
# done

for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=cd_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=cadmium_24x_random_simple" \
        scripts/gpu_job.sh
done
