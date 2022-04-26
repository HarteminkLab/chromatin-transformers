
import sys
sys.path.append('.')

import pickle
import os
import torch
import skimage
import cv2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from src.timer import Timer    
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from src.transformations import exhaustive_counts
from src.utils import write_pickle
from src.read_bam import read_mnase_bam


def get_mnase_img(gene, gene_mnase, window, resize_size, len_span):

    win_2 = window//2
    span = gene.TSS-win_2, gene.TSS+win_2

    cur_mnase = gene_mnase.copy()
    cur_mnase.mid = cur_mnase.mid - gene.TSS
    
    if gene.strand == '-':
        cur_mnase.mid = -cur_mnase.mid
    
    img = exhaustive_counts(cur_mnase, 
            (-win_2, win_2), len_span, x_key='mid', y_key='length')
    
    transform = Compose([ToPILImage(), Resize(resize_size), ToTensor()])

    img_t = transform(img.values.astype('float32').copy())[0]
    
    return img, img_t


def create_gene_image(gene, mnase, window, len_span, img_size):

    from src.chromatin import filter_mnase
    from src.vit_data import get_mnase_img

    win_2 = window//2

    span = gene.TSS-win_2, gene.TSS+win_2
    gene_mnase = filter_mnase(mnase, span[0], span[1], gene.chr, length_select=len_span)
    img, img_t = get_mnase_img(gene, gene_mnase, window, img_size, len_span)

    return img_t


def main():

    if len(sys.argv) < 2:
        raise ValueError("No BAM file specified")

    timer = Timer()

    bam_file = sys.argv[1]
    filename = bam_file.split('/')[-1].split('.')[0]

    print("Reading BAM...")
    mnase = read_mnase_bam(bam_file, timer=timer)
    orfs = pd.read_csv('data/orfs_cd_paper_dataset.csv').set_index('orf_name')

    img_size = 10, 100
    window = 1000
    len_span = (50, 200)

    imgs = np.zeros((len(orfs), img_size[0], img_size[1]))
    i = 0

    print("Generating MNase images...")
    saved_orfs = []
    for chrom in range(1, 17):
        timer.print_label(f"Chromosome {chrom}...")
        
        chrom_mnase = mnase[mnase.chr == chrom]
        chrom_orfs = orfs[orfs.chr == chrom]
        
        for orf_name, orf in chrom_orfs.iterrows():

            saved_orfs.append(orf_name)
            from src.vit_data import create_gene_image

            img_t = create_gene_image(orf, chrom_mnase, img_size=img_size,
                window=window, len_span=len_span)

            imgs[i] = img_t
            i += 1
            
            timer.print_progress(i, len(orfs), conditional=(i % 100 == 0))

    savepath = f'data/vit/vit_imgs_{filename}.pkl'
    save_tuple = ((f"img_size: {img_size}\n"
                   f"window: {window}\n"
                   f"lengths: {len_span}\n"
                   f"orfs: {saved_orfs}"), imgs)

    write_pickle(save_tuple, savepath)
    print(f"Done. Wrote to {savepath}")


if __name__ == '__main__':
    main()
