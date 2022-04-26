#!/bin/bash

declare -a BAMFILES=("/usr/xtmp/tqtran/data/cd/mnase/DM498_MNase_rep1_0_min.bam" 
				 	 "/usr/xtmp/tqtran/data/cd/mnase/DM499_MNase_rep1_7.5_min.bam"
				 	 "/usr/xtmp/tqtran/data/cd/mnase/DM500_MNase_rep1_15_min.bam"
				 	 "/usr/xtmp/tqtran/data/cd/mnase/DM501_MNase_rep1_30_min.bam"
				 	 "/usr/xtmp/tqtran/data/cd/mnase/DM502_MNase_rep1_60_min.bam"
				 	 "/usr/xtmp/tqtran/data/cd/mnase/DM503_MNase_rep1_120_min.bam"
				 	 "/usr/xtmp/tqtran/data/cd/mnase/DM504_MNase_rep2_0_min.bam"
				 	 "/usr/xtmp/tqtran/data/cd/mnase/DM505_MNase_rep2_7.5_min.bam"
				 	 "/usr/xtmp/tqtran/data/cd/mnase/DM506_MNase_rep2_15_min.bam"
				 	 "/usr/xtmp/tqtran/data/cd/mnase/DM507_MNase_rep2_30_min.bam"
				 	 "/usr/xtmp/tqtran/data/cd/mnase/DM508_MNase_rep2_60_min.bam"
				 	 "/usr/xtmp/tqtran/data/cd/mnase/DM509_MNase_rep2_120_min.bam")


# Iterate the string array using for loop
for BAM in ${BAMFILES[@]}; do
    sbatch -D ./slurm-logs/ --job-name="cd_data" --export="PYFILE=src/vit_data.py,ARGS=$BAM" scripts/cpu_job.sh
done
