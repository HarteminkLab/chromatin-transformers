# !/bin/bash

NUM_RUNS=5

# Data using merged replicates
#for RUN in `seq $NUM_RUNS`; do 
#    sbatch -D ./slurm-logs/ --job-name=24x_merged_$RUN \
#        --export="PYFILE=src/vit_train.py,ARGS=24x_120_merged" \
#        scripts/gpu_job.sh
#done


# Channel for each replicate
#for RUN in `seq $NUM_RUNS`; do 
#    sbatch -D ./slurm-logs/ --job-name=24x_repl_$RUN \
#        --export="PYFILE=src/vit_train.py,ARGS=24x_120_replicates" \
#        scripts/gpu_job.sh
#done

# Channel for each replicate
#for RUN in `seq $NUM_RUNS`; do 
#    sbatch -D ./slurm-logs/ --job-name=12x_merge_$RUN \
#        --export="PYFILE=src/vit_train.py,ARGS=12x_120_merged" \
#        scripts/gpu_job.sh
#done

# Simple 12x
# for RUN in `seq $NUM_RUNS`; do 
#     sbatch -D ./slurm-logs/ --job-name=12x_simpl_$RUN \
#         --export="PYFILE=src/vit_train.py,ARGS=12x_120_simple" \
#         scripts/gpu_job.sh
# done


# Complex 12x
# for RUN in `seq $NUM_RUNS`; do 
#     sbatch -D ./slurm-logs/ --job-name=12x_comp_$RUN \
#         --export="PYFILE=src/vit_train.py,ARGS=12x_120_complex" \
#         scripts/gpu_job.sh
# done

# Simple 24x
for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=24x_simpl_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=24x_120_simple" \
        scripts/gpu_job.sh
done


# Complex 24x
for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=24x_comp_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=24x_120_complex" \
        scripts/gpu_job.sh
done

