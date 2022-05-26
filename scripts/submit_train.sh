# !/bin/bash

NUM_RUNS=1

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

# Old data set with old training model
#sbatch -D ./slurm-logs/ --job-name=120_old \
#        --export="PYFILE=src/vit_train_old.py" \
#        scripts/gpu_job.sh

# Old data set with new training model
#sbatch -D ./slurm-logs/ --job-name=120_new \
#        --export="PYFILE=src/vit_train.py,ARGS=complex_32x128_120" \
#        scripts/gpu_job.sh

# New data set with new training model
for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=24x128_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=complex_24x128_120" \
        scripts/gpu_job.sh
done
