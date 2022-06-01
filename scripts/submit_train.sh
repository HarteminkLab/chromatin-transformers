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
for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=12x_merge_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=12x_120_merged" \
        scripts/gpu_job.sh
done
