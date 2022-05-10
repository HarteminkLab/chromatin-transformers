
import sys
sys.path.append('.')

import pandas as pd
import numpy as np

from src.timer import Timer    
from src.utils import write_pickle
from src.read_bam import read_rna_seq

from src.timer import Timer
from src.transcription import convert_to_TPM, calculate_read_counts


def convert_read_counts_TPM(read_counts_rep, times):

    TPMs = read_counts_rep[[]].copy()
    for time in times:
        TPMs.loc[:, time] = convert_to_TPM(read_counts_rep[time], orf['length'])
    return TPMs


def compute_replicate_read_counts(rna_bam_files, timer):

    orfs = pd.read_csv('data/orfs_cd_paper_dataset.csv').set_index('orf_name')
    read_counts = orfs[[]].copy()

    for time, bam_path in rna_bam_files:
        timer.print_label(f"{time}, reading BAM", end='...')
        rna_seq = read_rna_seq(bam_path, time)
        timer.print_label("Calculating read counts", end='...')
        cur_read_counts = calculate_read_counts(orfs, rna_seq)['count']
        read_counts.loc[:, time] = cur_read_counts
        print('Done.')

    return read_counts


def gen_rna():

    if len(sys.argv) < 2:
        raise ValueError("No BAM direcotry specified")

    parent_dir = sys.argv[1]

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

    timer = Timer()

    rep1_read_counts = compute_replicate_read_counts(rna_bam_files_rep1, timer)
    rep2_read_counts = compute_replicate_read_counts(rna_bam_files_rep2, timer)

    times = [0.0, 7.5, 15, 30, 60, 120]
    combined_read_counts = pd.concat([rep1_read_counts, rep2_read_counts]).groupby('orf_name').sum()

    TPM_values = convert_read_counts_TPM(combined_read_counts, times)

    save_path = 'data/vit/cd_rna_read_counts.csv'
    combined_read_counts.to_csv(save_path)
    print(f"Wrote to {save_path}")

    save_path = 'data/vit/cd_rna_seq_TPM.csv'
    TPM_values.to_csv(save_path)
    print(f"Wrote to {save_path}")


if __name__ == '__main__':
    gen_rna()
