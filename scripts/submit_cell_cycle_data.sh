#!/bin/bash

MNASE_DIRECTORY= # TODO: set path to mnase data

declare -a BAMFILES=("$MNASE_DIRECTORY/DMAH64_MNase_rep1_0_min.bam"
                     "$MNASE_DIRECTORY/DMAH66_MNase_rep1_20_min.bam"
                     "$MNASE_DIRECTORY/DMAH67_MNase_rep1_30_min.bam"
                     "$MNASE_DIRECTORY/DMAH68_MNase_rep1_40_min.bam"
                     "$MNASE_DIRECTORY/DMAH69_MNase_rep1_50_min.bam"
                     "$MNASE_DIRECTORY/DMAH70_MNase_rep1_60_min.bam"
                     "$MNASE_DIRECTORY/DMAH71_MNase_rep1_70_min.bam"
                     "$MNASE_DIRECTORY/DMAH72_MNase_rep1_80_min.bam"
                     "$MNASE_DIRECTORY/DMAH73_MNase_rep1_90_min.bam"
                     "$MNASE_DIRECTORY/DMAH74_MNase_rep1_100_min.bam"
                     "$MNASE_DIRECTORY/DMAH75_MNase_rep1_110_min.bam"
                     "$MNASE_DIRECTORY/DMAH76_MNase_rep1_120_min.bam"
                     "$MNASE_DIRECTORY/DMAH77_MNase_rep1_130_min.bam"
                     "$MNASE_DIRECTORY/DMAH78_MNase_rep1_140_min.bam"
                     "$MNASE_DIRECTORY/DMAH79_MNase_rep1_150_min.bam"
                     
                     "$MNASE_DIRECTORY/DMAH82_MNase_rep2_0_min.bam"
                     "$MNASE_DIRECTORY/DMAH83_MNase_rep2_10_min.bam"
                     "$MNASE_DIRECTORY/DMAH84_MNase_rep2_20_min.bam"
                     "$MNASE_DIRECTORY/DMAH85_MNase_rep2_30_min.bam"
                     "$MNASE_DIRECTORY/DMAH86_MNase_rep2_40_min.bam"
                     "$MNASE_DIRECTORY/DMAH87_MNase_rep2_50_min.bam"
                     "$MNASE_DIRECTORY/DMAH88_MNase_rep2_60_min.bam"
                     "$MNASE_DIRECTORY/DMAH89_MNase_rep2_70_min.bam"
                     "$MNASE_DIRECTORY/DMAH90_MNase_rep2_80_min.bam"
                     "$MNASE_DIRECTORY/DMAH91_MNase_rep2_90_min.bam"
                     "$MNASE_DIRECTORY/DMAH92_MNase_rep2_100_min.bam"
                     "$MNASE_DIRECTORY/DMAH93_MNase_rep2_110_min.bam"
                     "$MNASE_DIRECTORY/DMAH94_MNase_rep2_120_min.bam"
                     "$MNASE_DIRECTORY/DMAH95_MNase_rep2_130_min.bam"
                     "$MNASE_DIRECTORY/DMAH96_MNase_rep2_140_min.bam")

OUT_DIR='data/vit/cell_cycle_24x128_p1/'

i=1
for BAM in ${BAMFILES[@]}; do
   sbatch -D ./slurm-logs/ --job-name="cc_$i" --export="PYFILE=src/vit_img_gen.py,ARGS=$BAM $OUT_DIR" scripts/cpu_job.sh
   ((i=i+1))
done

#RNA_DIR="/usr/xtmp/tqtran/data/cell_cycle/rna/"
#sbatch -D ./slurm-logs/ --job-name="vrna_$i" --export="PYFILE=src/vit_rna_gen.py,ARGS=$RNA_DIR" scripts/cpu_job.sh
