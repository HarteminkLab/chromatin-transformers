#!/bin/bash

declare -a BAMFILES=("/usr/xtmp/tqtran/data/cd/mnase/DM498_MNase_rep1_0_min.bam")

# Iterate the string array using for loop
for BAM in ${BAMFILES[@]}; do
    sbatch -D ./slurm-logs/ --job-name="cd_data" --export="PYFILE=src/vit_data.py,ARGS=$BAM" scripts/cpu_job.sh
done

