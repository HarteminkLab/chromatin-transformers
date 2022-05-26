# !/bin/bash

NUM_RUNS=5

# for RUN in `seq $NUM_RUNS`; do 
#     sbatch -D ./slurm-logs/ --job-name=s_$RUN \
#         --export="PYFILE=src/vit_train.py,ARGS=simple" \
#         scripts/gpu_job.sh
# done

# for RUN in `seq $NUM_RUNS`; do 
#     sbatch -D ./slurm-logs/ --job-name=c_$RUN \
#         --export="PYFILE=src/vit_train.py,ARGS=complex" \
#         scripts/gpu_job.sh
# done

# for RUN in `seq $NUM_RUNS`; do 
#     sbatch -D ./slurm-logs/ --job-name=t_$RUN \
#         --export="PYFILE=src/vit_train.py,ARGS=complex_120" \
#         scripts/gpu_job.sh
# done


sbatch -D ./slurm-logs/ --job-name=120_old \
        --export="PYFILE=src/vit_train_old.py" \
        scripts/gpu_job.sh

sbatch -D ./slurm-logs/ --job-name=120_new \
        --export="PYFILE=src/vit_train.py,ARGS=complex_32x128_120" \
        scripts/gpu_job.sh
