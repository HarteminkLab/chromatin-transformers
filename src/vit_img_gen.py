
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
from src.transformations import exhaustive_counts
from src.utils import write_pickle
from src.read_bam import read_mnase_bam
from src.utils import normal_2d_kernel
from src.chromatin import filter_mnase
from src.utils import print_fl, mkdir_safe


class ViTImgGen:

    def __init__(self, mnase, window, sublength_resize_height, len_cuts, img_width, patch_size):

        self.mnase = mnase
        self.window = window
        self.sublength_resize_height = sublength_resize_height
        self.len_cuts = len_cuts
        self.img_width = img_width
        self.len_span = len_cuts[0], len_cuts[-1]
        self.patch_size = patch_size

        # Computed attributes
        self.column_patches = img_width / patch_size
        self.patch_width_bp = window / self.column_patches


    def get_mnase_img(self, gene):

        from scipy.signal import convolve2d

        window = self.window 
        sublength_resize_height = self.sublength_resize_height 
        len_cuts = self.len_cuts 
        len_span = self.len_span
        img_width = self.img_width

        # Convert reads to a count matrix
        win_2 = window//2
        span = gene.TSS-win_2, gene.TSS+win_2
        gene_mnase = filter_mnase(self.mnase, span[0], span[1], gene.chr)

        img = exhaustive_counts(gene_mnase, 
                (span[0], span[1]), len_span, x_key='mid', y_key='length')

        # Smooth the source image
        kernel = normal_2d_kernel(5, 15, 20, 30)
        smoothed = convolve2d(img.loc[len_span[0]:len_span[1]], kernel, mode='same')
        lens = np.arange(len_span[0], len_span[1])
        smoothed = pd.DataFrame(smoothed, index=lens, columns=np.arange(-win_2, win_2))
        scaled_img, img_slices = partition_and_resize(smoothed, len_cuts, sublength_resize_height, 
            img_width)

        self.img, self.scaled_img, self.smoothed, self.img_slices = img, scaled_img, smoothed, img_slices

        return img, scaled_img, smoothed, img_slices


    def plot_resized_img(self):

        img, img_t, smoothed, img_slices = self.img, self.scaled_img, self.smoothed, self.img_slices
        win_2 = self.window//2
        len_span = self.len_span
        len_cuts = self.len_cuts

        def plot_len_cuts():
            for y in len_cuts:
                plt.axhline(y, lw=1, c='gray', linestyle='dotted')
            plt.yticks(len_cuts)

        def plot_len_cuts_scaled():

            for y in np.arange(0, 12, 4):
                plt.axhline(y, lw=1, c='gray', linestyle='dotted')

        def plot_xpatches():
            for x in np.arange(-win_2, win_2, self.patch_width_bp):
                plt.axvline(x, lw=1, c='gray', linestyle='dotted')
            plt.axvline(0, c='black', lw=1)

        def plot_img(img, extent=[-win_2, win_2, 0, 225]):
            plt.imshow(img, cmap='magma_r', vmax=0.25, origin='lower', 
                extent=extent, aspect='auto', interpolation='none')
            plt.xticks([])
            plt.ylim(20, 225)

        plt.figure(figsize=(7, 6.25))

        plt.subplot(4, 1, 1)
        plot_img(img, extent=[-win_2, win_2, *len_span])
        plot_len_cuts()
        plot_xpatches()

        plt.subplot(4, 1, 2)
        plot_img(smoothed, extent=[-win_2, win_2, *len_span])
        plot_len_cuts()
        plot_xpatches()

        plt.subplot(4, 1, 3)
        plot_scaled(img_t, self.window)
        plt.xticks([])
        plot_xpatches()
        plot_len_cuts_scaled()

        plt.subplot(4, 1, 4)
        rescaled_t = cv2.resize(img_t, (self.window, len_cuts[-1]-len_cuts[0]))
        plot_scaled(rescaled_t, self.window)
        plt.xticks(np.arange(-400, 600, 200))
        plot_xpatches()
        plot_len_cuts_scaled()


def subselect_resize(smoothed, len_subselect, resize_size):
    smooth_sub = smoothed.loc[len_subselect[0]:len_subselect[1]-1]
    img_t = cv2.resize(smooth_sub.values, (resize_size[1], resize_size[0]))
    return smooth_sub, img_t


def partition_and_resize(smoothed, len_cuts, sublen_size, pos_resize):

    img_slices = []

    scaled_img = np.zeros(((sublen_size*(len(len_cuts)-1)), pos_resize))
    for i in range(1, len(len_cuts)):
        len_subselect = (len_cuts[i-1], len_cuts[i])
        smooth_sub, img_t = subselect_resize(smoothed, len_subselect, (sublen_size, pos_resize))
        img_slices.append((len_subselect, smooth_sub, img_t))
        scaled_img[((i-1)*sublen_size):(i*sublen_size)] = img_t
        
    return scaled_img, img_slices


def plot_scaled(scaled_img, window):

    win_2 = window//2

    plt.imshow(scaled_img, vmax=0.25, cmap='magma_r', origin='lower', aspect='auto',
           extent=[-win_2, win_2, 0, 12], interpolation='none')

    plt.yticks([2, 6, 10], ['Small', 'Intermediate', 'Nucleosomal'])
    plt.ylim(0, 12)


def subselect_resize(smoothed, len_subselect, resize_size):
    smooth_sub = smoothed.loc[len_subselect[0]:len_subselect[1]-1]
    img_t = cv2.resize(smooth_sub.values, (resize_size[1], resize_size[0]))
    return smooth_sub, img_t


def main():

    if len(sys.argv) < 3:
        raise ValueError("No BAM file specified")

    timer = Timer()

    bam_file = sys.argv[1]
    out_dir = sys.argv[2]
    filename = bam_file.split('/')[-1].split('.')[0]

    mkdir_safe(out_dir)

    print("Reading BAM...")
    mnase = read_mnase_bam(bam_file, timer=timer)
    orfs = pd.read_csv('data/orfs_cd_paper_dataset.csv').set_index('orf_name')

    # Partitions lengths of fragments into equal sized (length-wise) patches
    # Small fragments: 30-79
    # Intermediate fragments: 80-129
    # Nucleosomal fragments: 130-200
    len_cuts = [30, 80, 130, 201]

    cuts = len(len_cuts)-1 # 3
    window = 1024
    
    img_height = 96
    img_width = 512

    patch_size = img_height // cuts
    sublength_resize_height = patch_size # times 3 vertical patches of height

    vit_gen = ViTImgGen(mnase, window, sublength_resize_height, len_cuts,
                        img_width, patch_size)
    imgs = np.zeros((len(orfs), img_height, img_width))
    i = 0

    print("Generating MNase images...")
    saved_orfs = []
    chroms = []
    for chrom in range(1, 17):
        timer.print_label(f"Chromosome {chrom}...")
        
        chrom_mnase = mnase[mnase.chr == chrom]
        chrom_orfs = orfs[orfs.chr == chrom]
        
        for orf_name, orf in chrom_orfs.iterrows():

            saved_orfs.append(orf_name)
            chroms.append(chrom)
            img, img_t, smoothed, img_slices = vit_gen.get_mnase_img(orf)
            imgs[i] = img_t
            i += 1
            
            timer.print_progress(i, len(orfs), conditional=(i % 100 == 0))

    # Insert channel dim
    imgs = imgs.reshape(imgs.shape[0], 1, imgs.shape[1], imgs.shape[2])

    desc_dict = {"img_size": (img_height, img_width),
                 "window": window,
                 "length_cuts": len_cuts,
                 "patch_size": patch_size,
                 "img_height": img_height,
                 "img_width": img_width,
                 "chrs": chroms,
                 "lengths": vit_gen.len_span,
                 "orfs": saved_orfs}

    savepath = f'data/vit/vit_imgs_{img_height}x{img_width}_{filename}.pkl'
    save_tuple = (desc_dict, imgs)

    write_pickle(save_tuple, savepath)
    print(f"Done. Wrote to {savepath}")


if __name__ == '__main__':
    main()
