
import sys
sys.path.append('.')

import pandas as pd
import numpy as np

from src.timer import Timer    
from src.utils import write_pickle
from src.read_bam import read_rna_seq

from src.timer import Timer
from src.transcription import convert_to_TPM, calculate_read_counts


def convert_read_counts_TPM(read_counts_rep, orfs, times):

    TPMs = read_counts_rep[[]].copy()
    for time in times:
        TPMs.loc[:, time] = convert_to_TPM(read_counts_rep[time], orfs['length'])
    return TPMs


def compute_replicate_read_counts(rna_bam_files, orfs, timer):

    read_counts = orfs[[]].copy()

    for time, bam_path in rna_bam_files:
        timer.print_label(f"{time}, reading BAM", end='...')
        rna_seq = read_rna_seq(bam_path, time)
        timer.print_label("Calculating read counts", end='...')
        cur_read_counts = calculate_read_counts(orfs, rna_seq)['count']
        read_counts.loc[:, time] = cur_read_counts
        print('Done.')

    return read_counts


def gen_cell_cycle_rna(parent_dir):

    rna_bam_files_rep1 = [(0, f'{parent_dir}/DMAH1_RNA_rep1_0_min.bam'),
                          (20, f'{parent_dir}/DMAH3_RNA_rep1_20_min.bam'),
                          (30, f'{parent_dir}/DMAH4_RNA_rep1_30_min.bam'),
                          (40, f'{parent_dir}/DMAH5_RNA_rep1_40_min.bam'),
                          (50, f'{parent_dir}/DMAH6_RNA_rep1_50_min.bam'),
                          (60, f'{parent_dir}/DMAH7_RNA_rep1_60_min.bam'),
                          (70, f'{parent_dir}/DMAH8_RNA_rep1_70_min.bam'),
                          (80, f'{parent_dir}/DMAH9_RNA_rep1_80_min.bam'),
                          (90, f'{parent_dir}/DMAH10_RNA_rep1_90_min.bam'),
                          (100, f'{parent_dir}/DMAH31_RNA_rep1_100_min.bam'),
                          (120, f'{parent_dir}/DMAH33_RNA_rep1_120_min.bam'),
                          (130, f'{parent_dir}/DMAH34_RNA_rep1_130_min.bam'),
                          (140, f'{parent_dir}/DMAH35_RNA_rep1_140_min.bam')]

    rna_bam_files_rep2 = [(0, f'{parent_dir}/DMAH100_RNA_rep2_0_min.bam'),
                          (20, f'{parent_dir}/DMAH102_RNA_rep2_20_min.bam'),
                          (30, f'{parent_dir}/DMAH103_RNA_rep2_30_min.bam'),
                          (40, f'{parent_dir}/DMAH104_RNA_rep2_40_min.bam'),
                          (50, f'{parent_dir}/DMAH105_RNA_rep2_50_min.bam'),
                          (60, f'{parent_dir}/DMAH106_RNA_rep2_60_min.bam'),
                          (70, f'{parent_dir}/DMAH107_RNA_rep2_70_min.bam'),
                          (80, f'{parent_dir}/DMAH108_RNA_rep2_80_min.bam'),
                          (90, f'{parent_dir}/DMAH109_RNA_rep2_90_min.bam'),
                          (100, f'{parent_dir}/DMAH110_RNA_rep2_100_min.bam'),
                          (120, f'{parent_dir}/DMAH112_RNA_rep2_120_min.bam'),
                          (130, f'{parent_dir}/DMAH113_RNA_rep2_130_min.bam'),
                          (140, f'{parent_dir}/DMAH114_RNA_rep2_140_min.bam')]

    times = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 140]
    gen_rna(parent_dir, rna_bam_files_rep1, rna_bam_files_rep2, times, 'data/vit', 'cell_cycle_rna')


def gen_cd_rna(parent_dir):
    rna_bam_files_rep1 = [(0.0, f'{parent_dir}/DM538_RNA_rep1_0_min.bam'),
                          (7.5, f'{parent_dir}/DM539_RNA_rep1_7.5_min.bam'),
                          (15.0, f'{parent_dir}/DM540_RNA_rep1_15_min.bam'),
                          (30.0, f'{parent_dir}/DM541_RNA_rep1_30_min.bam'),
                          (60.0, f'{parent_dir}/DM542_RNA_rep1_60_min.bam'),
                          (120.0, f'{parent_dir}/DM543_RNA_rep1_120_min.bam')]

    rna_bam_files_rep2 = [(0.0, f'{parent_dir}/DM1450_RNA_rep2_0_min.bam'),
                          (7.5, f'{parent_dir}/DM1451_RNA_rep2_7.5_min.bam'),
                          (15.0, f'{parent_dir}/DM1452_RNA_rep2_15_min.bam'),
                          (30.0, f'{parent_dir}/DM1453_RNA_rep2_30_min.bam'),
                          (60.0, f'{parent_dir}/DM1454_RNA_rep2_60_min.bam'),
                          (120.0, f'{parent_dir}/DM1455_RNA_rep2_120_min.bam')]

    times = [0.0, 7.5, 15, 30, 60, 120]
    gen_rna(parent_dir, rna_bam_files_rep1, rna_bam_files_rep2, times, 'data/vit', 'cd_rna')


def gen_rna(parent_dir, rna_bam_files_rep1, rna_bam_files_rep2, times, save_dir, save_prefix):

    timer = Timer()

    # Fixed dataset from cadmium paper
    orfs = pd.read_csv('data/orfs_cd_paper_dataset.csv').set_index('orf_name')

    rep1_read_counts = compute_replicate_read_counts(rna_bam_files_rep1, orfs, timer)
    rep2_read_counts = compute_replicate_read_counts(rna_bam_files_rep2, orfs, timer)

    combined_read_counts = pd.concat([rep1_read_counts, rep2_read_counts]).groupby('orf_name').sum()
    TPM_values = convert_read_counts_TPM(combined_read_counts, orfs, times)

    save_path = f"{save_dir}/{save_path}_read_counts.csv"
    combined_read_counts.to_csv(save_path)
    print(f"Wrote to {save_path}")

    save_path = f"{save_dir}/{save_path}_TPM.csv"
    TPM_values.to_csv(save_path)
    print(f"Wrote to {save_path}")
    timer.print_time()


def main():

    if len(sys.argv) < 2:
        raise ValueError("No BAM directory specified")

    parent_dir = sys.argv[1]
    gen_cell_cycle_rna(parent_dir)


if __name__ == '__main__':
    main()
