# !/bin/bash

NUM_RUNS=2

for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=cc_simple_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=cell_cycle_24x_chr4_validchrom_b4" \
        scripts/gpu_job.sh
done

for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=cc_simple_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=cell_cycle_24x_chr4_validchrom_b4_simple" \
        scripts/gpu_job.sh
done

for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=cc_simple_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=cell_cycle_24x_chr4_validchrom_b32_simple" \
        scripts/gpu_job.sh
done

for RUN in `seq $NUM_RUNS`; do 
    sbatch -D ./slurm-logs/ --job-name=cc_simple_$RUN \
        --export="PYFILE=src/vit_train.py,ARGS=cell_cycle_24x_chr4_validchrom_b96_simple" \
        scripts/gpu_job.sh
done
