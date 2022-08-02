# !/bin/bash

## Resume long running

# sbatch -D ./slurm-logs/ --job-name=lr_$RUN \
#     --export="PYFILE=src/vit_train.py,ARGS=resume output/cd_24x128_chr2_long_running_20220715_a100" \
#     scripts/gpu_job.sh

# sbatch -D ./slurm-logs/ --job-name=lr_$RUN \
#     --export="PYFILE=src/vit_train.py,ARGS=resume output/cd_24x128_chr2_long_running_20220715_b763" \
#     scripts/gpu_job.sh

# sbatch -D ./slurm-logs/ --job-name=lr_$RUN \
#     --export="PYFILE=src/vit_train.py,ARGS=resume output/cd_24x128_chr2_long_running_20220715_cb94" \
#     scripts/gpu_job.sh

NUM_RUNS=5

for RUN in `seq $NUM_RUNS`; do 
   sbatch -D ./slurm-logs/ --job-name=p1_$RUN \
       --export="PYFILE=src/vit_train.py,ARGS=cd_24x_p1_more_complex" \
       scripts/gpu_job.sh
done
