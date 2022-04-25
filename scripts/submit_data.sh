#!/bin/bash

sbatch -D ./slurm-logs/ --job-name="cd_data" --export="PYFILE=src/vit_data.py" scripts/cpu_job.sh
