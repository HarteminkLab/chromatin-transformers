# !/bin/bash


sbatch -D ./slurm-logs/ --job-name=lr_$RUN \
    --export="ARGS=output/cd_24x128_chr2_long_running_20220715_a100 resume" \
    scripts/gpu_job.sh

sbatch -D ./slurm-logs/ --job-name=lr_$RUN \
    --export="ARGS=output/cd_24x128_chr2_long_running_20220715_b763 resume" \
    scripts/gpu_job.sh

sbatch -D ./slurm-logs/ --job-name=lr_$RUN \
    --export="ARGS=output/cd_24x128_chr2_long_running_20220715_cb94 resume" \
    scripts/gpu_job.sh
