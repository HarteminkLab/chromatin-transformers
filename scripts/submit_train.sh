# !/bin/bash

NUM_RUNS=2

# for RUN in `seq $NUM_RUNS`; do 
#    sbatch -D ./slurm-logs/ --job-name=sim_$RUN \
#        --export="PYFILE=src/vit_train.py,ARGS=cadmium_24x_random_simple" \
#        scripts/gpu_job.sh
# done

# for RUN in `seq $NUM_RUNS`; do 
#    sbatch -D ./slurm-logs/ --job-name=bas_$RUN \
#        --export="PYFILE=src/vit_train.py,ARGS=cadmium_24x_random_baseline" \
#        scripts/gpu_job.sh
# done

# for RUN in `seq $NUM_RUNS`; do 
#    sbatch -D ./slurm-logs/ --job-name=com_$RUN \
#        --export="PYFILE=src/vit_train.py,ARGS=cadmium_24x_random_complex" \
#        scripts/gpu_job.sh
# done

for RUN in `seq $NUM_RUNS`; do 
   sbatch -D ./slurm-logs/ --job-name=96x_$RUN \
       --export="PYFILE=src/vit_train.py,ARGS=cadmium_96x_random_complex" \
       scripts/gpu_job.sh
done
