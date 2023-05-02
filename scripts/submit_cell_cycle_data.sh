#!/bin/bash

declare -a BAMFILES=("/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH64_MNase_rep1_0_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH66_MNase_rep1_20_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH67_MNase_rep1_30_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH68_MNase_rep1_40_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH69_MNase_rep1_50_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH70_MNase_rep1_60_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH71_MNase_rep1_70_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH72_MNase_rep1_80_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH73_MNase_rep1_90_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH74_MNase_rep1_100_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH75_MNase_rep1_110_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH76_MNase_rep1_120_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH77_MNase_rep1_130_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH78_MNase_rep1_140_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH79_MNase_rep1_150_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH82_MNase_rep2_0_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH83_MNase_rep2_10_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH84_MNase_rep2_20_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH85_MNase_rep2_30_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH86_MNase_rep2_40_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH87_MNase_rep2_50_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH88_MNase_rep2_60_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH89_MNase_rep2_70_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH90_MNase_rep2_80_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH91_MNase_rep2_90_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH92_MNase_rep2_100_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH93_MNase_rep2_110_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH94_MNase_rep2_120_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH95_MNase_rep2_130_min.bam"
                     "/usr/xtmp/tqtran/data/cell_cycle/mnase/DMAH96_MNase_rep2_140_min.bam")

OUT_DIR='data/vit/cell_cycle_24x128_p1/'

i=1
for BAM in ${BAMFILES[@]}; do
   sbatch -D ./slurm-logs/ --job-name="cc_$i" --export="PYFILE=src/vit_img_gen.py,ARGS=$BAM $OUT_DIR" scripts/cpu_job.sh
   ((i=i+1))
done

#RNA_DIR="/usr/xtmp/tqtran/data/cell_cycle/rna/"
#sbatch -D ./slurm-logs/ --job-name="vrna_$i" --export="PYFILE=src/vit_rna_gen.py,ARGS=$RNA_DIR" scripts/cpu_job.sh
