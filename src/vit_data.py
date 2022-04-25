
import sys
sys.path.append('.')

import pickle
import os
import torch
import skimage
import cv2

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from src.transformations import exhaustive_counts
from src.utils import write_pickle


def get_mnase_img(gene, gene_mnase, time, window, resize_size, len_span):

    win_2 = window//2
    span = gene.TSS-win_2, gene.TSS+win_2

    cur_mnase = gene_mnase.copy()
    cur_mnase.mid = cur_mnase.mid - gene.TSS
    
    if gene.strand == '-':
        cur_mnase.mid = -cur_mnase.mid
    
    img = exhaustive_counts(cur_mnase[cur_mnase.time == time], 
            (-win_2, win_2), len_span, x_key='mid', y_key='length')
    
    transform = Compose([ToPILImage(), Resize(resize_size), ToTensor()])

    img_t = transform(img.values.astype('float32').copy())[0]
    
    return img, img_t


def create_gene_image(gene, mnase, time, window=1000, len_span=(50, 200), img_size=(10, 100)):

    from src.chromatin import filter_mnase
    from src.vit_data import get_mnase_img

    win_2 = window//2

    span = gene.TSS-win_2, gene.TSS+win_2
    gene_mnase = filter_mnase(mnase, span[0], span[1], gene.chr, length_select=len_span)
    img, img_t = get_mnase_img(gene, gene_mnase, time, window, img_size, len_span)

    return img_t



def main():

    mnase = pd.read_hdf('data/mnase_seq_merged_sampled.h5.z')
    orfs = pd.read_csv('data/orfs_cd_paper_dataset.csv').set_index('orf_name')

    from src.timer import Timer

    img_size = 10, 100
    

    timer = Timer()

    times = mnase.time.unique()

    for time in times:

        timer.print_label(f"Time {time}...")

        imgs = np.zeros((len(orfs), img_size[0], img_size[1]))
        i = 0

        for chrom in range(1, 17):
            timer.print_label(f"Chromosome {chrom}...")
            
            chrom_mnase = mnase[mnase.chr == chrom]
            chrom_orfs = orfs[orfs.chr == chrom]
            
            for orf_name, orf in chrom_orfs.iterrows():
                from src.vit_data import create_gene_image
                img_t = create_gene_image(orf, chrom_mnase, time, img_size=img_size)
                imgs[i] = img_t
                i += 1
                
                timer.print_progress(i, len(orfs), conditional=(i % 100 == 0))

        write_pickle(imgs, 'data/mnase_10x100_cd_{}.pkl')


if __name__ == '__main__':
    main()
