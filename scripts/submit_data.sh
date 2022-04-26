#!/bin/bash




BAM=

sbatch -D ./slurm-logs/ --job-name="cd_data" --export="PYFILE=src/vit_data.py,ARGS=$BAM" scripts/cpu_job.sh
