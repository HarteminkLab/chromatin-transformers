
import sys
sys.path.append('.')

from src.timer import Timer
from src.read_bam import read_mnase_bam
import pandas as pd
from config.data_gen import data24x128 as data_config
from src.utils import print_fl
from src.vit_img_gen import ViTImgGen


def main():
 
    directory = '/usr/xtmp/tqtran/data/cell_cycle/mnase/'
    # directory = '/Users/trung/Research/data/bam/cell_cycle/mnase'
    out_dir = 'data/vit/cell_cycle_24x128_p1/'

    bam_files = [f"{directory}/DMAH64_MNase_rep1_0_min.bam",
                 f"{directory}/DMAH66_MNase_rep1_20_min.bam",
                 f"{directory}/DMAH67_MNase_rep1_30_min.bam",
                 f"{directory}/DMAH68_MNase_rep1_40_min.bam",
                 f"{directory}/DMAH69_MNase_rep1_50_min.bam",
                 f"{directory}/DMAH70_MNase_rep1_60_min.bam",
                 f"{directory}/DMAH71_MNase_rep1_70_min.bam",
                 f"{directory}/DMAH72_MNase_rep1_80_min.bam",
                 f"{directory}/DMAH73_MNase_rep1_90_min.bam",
                 f"{directory}/DMAH74_MNase_rep1_100_min.bam",
                 f"{directory}/DMAH75_MNase_rep1_110_min.bam",
                 f"{directory}/DMAH76_MNase_rep1_120_min.bam",
                 f"{directory}/DMAH77_MNase_rep1_130_min.bam",
                 f"{directory}/DMAH78_MNase_rep1_140_min.bam",
                 f"{directory}/DMAH79_MNase_rep1_150_min.bam",
                 f"{directory}/DMAH82_MNase_rep2_0_min.bam",
                 f"{directory}/DMAH83_MNase_rep2_10_min.bam",
                 f"{directory}/DMAH84_MNase_rep2_20_min.bam",
                 f"{directory}/DMAH85_MNase_rep2_30_min.bam",
                 f"{directory}/DMAH86_MNase_rep2_40_min.bam",
                 f"{directory}/DMAH87_MNase_rep2_50_min.bam",
                 f"{directory}/DMAH88_MNase_rep2_60_min.bam",
                 f"{directory}/DMAH89_MNase_rep2_70_min.bam",
                 f"{directory}/DMAH90_MNase_rep2_80_min.bam",
                 f"{directory}/DMAH91_MNase_rep2_90_min.bam",
                 f"{directory}/DMAH92_MNase_rep2_100_min.bam",
                 f"{directory}/DMAH93_MNase_rep2_110_min.bam",
                 f"{directory}/DMAH94_MNase_rep2_120_min.bam",
                 f"{directory}/DMAH95_MNase_rep2_130_min.bam",
                 f"{directory}/DMAH96_MNase_rep2_140_min.bam",
    ]

    timer = Timer()

    orfs = pd.read_csv('data/orfs_cd_paper_dataset.csv').set_index('orf_name')
    p1_positions = orfs[['chr']].copy()
    p1_positions['p1_position'] = 0

    # Configuration for cutting the input MNase image
    len_cuts = data_config.LEN_CUTS
    cuts = len(len_cuts)-1 # 3
    window = data_config.WINDOW
    img_height = data_config.IMG_HEIGHT
    img_width = data_config.IMG_WIDTH
    patch_size = img_height // cuts
    sublength_resize_height = patch_size # times 3 vertical patches of height

    for chrom in range(1, 17):

        print_fl(f"Chromosome {chrom}")
        chrom_orfs = orfs[orfs.chr == chrom]

        # Load the MNase data for the chromosome
        # by combining all MNase data from input bamfile sample
        i = 0
        collect_mnase = pd.DataFrame()
        print_fl(f" Loading MNase data across {len(bam_files)} bam files")
        for bam_file in bam_files:
            mnase = read_mnase_bam(bam_file, chroms=[chrom])
            chrom_mnase = mnase[mnase.chr == chrom]
            collect_mnase = pd.concat([collect_mnase, chrom_mnase])
            timer.print_progress(i, len(bam_files), every=4, indent=4)
            i+=1

        # Create image generator
        vit_gen = ViTImgGen(collect_mnase, window, sublength_resize_height, len_cuts,
                            img_width, patch_size)
            
        # For each orf, compute the +1 location from the combined MNase-seq data
        i = 0
        print_fl(f" Computing +1 location for {len(chrom_orfs)} ORFs")
        for orf_name, orf in chrom_orfs.iterrows():
            img, img_t, smoothed, img_slices, p1_pos = vit_gen.get_mnase_img(orf)
            p1_positions.loc[orf_name, 'p1_position'] = p1_pos
            timer.print_progress(i, len(chrom_orfs), every=10, indent=4)
            i+=1

        p1_positions.to_csv(f"{out_dir}/computed_+1_positions.csv")

if __name__ == '__main__':
    main()
