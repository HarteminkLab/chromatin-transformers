#!/bin/bash

MNASE_DIRECTORY= # TODO: set path to mnase data

declare -a BAMFILES=("$MNASE_DIRECTORY/DM498_MNase_rep1_0_min.bam" 
				 	 "$MNASE_DIRECTORY/DM499_MNase_rep1_7.5_min.bam"
				 	 "$MNASE_DIRECTORY/DM500_MNase_rep1_15_min.bam"
				 	 "$MNASE_DIRECTORY/DM501_MNase_rep1_30_min.bam"
				 	 "$MNASE_DIRECTORY/DM502_MNase_rep1_60_min.bam"
				 	 "$MNASE_DIRECTORY/DM503_MNase_rep1_120_min.bam"
				 	 "$MNASE_DIRECTORY/DM504_MNase_rep2_0_min.bam"
				 	 "$MNASE_DIRECTORY/DM505_MNase_rep2_7.5_min.bam"
				 	 "$MNASE_DIRECTORY/DM506_MNase_rep2_15_min.bam"
				 	 "$MNASE_DIRECTORY/DM507_MNase_rep2_30_min.bam"
				 	 "$MNASE_DIRECTORY/DM508_MNase_rep2_60_min.bam"
				 	 "$MNASE_DIRECTORY/DM509_MNase_rep2_120_min.bam")

OUTDIR="data/vit/cd/24x128_p1"

i=1
for BAM in ${BAMFILES[@]}; do
    sbatch -D ./slurm-logs/ --job-name="vdat_$i" --export="PYFILE=src/vit_img_gen.py,ARGS=$BAM $OUTDIR" scripts/cpu_job.sh
    ((i=i+1))
done

#RNA_DIR="/usr/xtmp/tqtran/data/cd/rna/"
#sbatch -D ./slurm-logs/ --job-name="vrna_$i" --export="PYFILE=src/vit_rna_gen.py,ARGS=$RNA_DIR" scripts/cpu_job.sh

