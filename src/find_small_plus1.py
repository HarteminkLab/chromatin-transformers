
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
from scipy.signal import convolve2d


def find_small_prom(cur_data, plot=True):
    
    window_size = 1024
    window_size_2 = window_size//2
    scale = window_size // cur_data.shape[1]
    nuc_search_span = -160, 160

    if plot:
        plt.figure(figsize=(8, 2))
        plt.imshow(cur_data, origin='lower', cmap='magma_r', aspect='auto',
            extent=[-window_size_2, window_size_2, -0.5, 2.5])
        plt.show()

    kernel = normal_2d_kernel(1, 1, 5, 5)
    smoothed = convolve2d(cur_data, kernel, mode='same')    

    if plot:
        plt.figure(figsize=(8, 2))
        plt.imshow(smoothed, origin='lower', cmap='magma_r', aspect='auto',
            extent=[-window_size_2, window_size_2, -0.5, 2.5])
        plt.show()

    small_window = smoothed[:8]

    if plot:
        plt.figure(figsize=(8, 2))
        plt.imshow(small_window, origin='lower', cmap='magma_r', aspect='auto',
            extent=[-window_size_2, window_size_2, 0, 1])
        plt.yticks([])
        plt.show()

    small_data = small_window.sum(axis=0)
    occ_df = pd.DataFrame(index=np.arange(-64, 64)*scale, data={'occupancy': small_data})

    if plot:
        plt.figure(figsize=(8, 2))
        plt.plot(occ_df)
        plt.yticks([])

    search_span_idx = (nuc_search_span[0]),(nuc_search_span[1])
    found_index = occ_df.loc[search_span_idx[0]:search_span_idx[1]].idxmax()

    if plot:
        plt.axvline(found_index.values[0], c='red')

    return found_index.values[0]


def find_p1(cur_data, nuc_len_span):
    
    window_size = cur_data.shape[1]
    window_size_2 = window_size//2
    nuc_search_span = -160, 160

    nuc_window = cur_data.loc[nuc_len_span[0]:nuc_len_span[1]].values

    nuc_data = nuc_window.sum(axis=0)
    occ_df = pd.DataFrame(index=cur_data.columns, data={'occupancy': nuc_data})

    search_span_idx = (nuc_search_span[0]),(nuc_search_span[1])
    found_index = occ_df.loc[search_span_idx[0]:search_span_idx[1]].idxmax()

    return found_index.values[0]


def shift_for_p1(smoothed, vit_gen):
    nuc_len_span = vit_gen.len_cuts[-2], vit_gen.len_cuts[-1]

    # Find the +1 nucleosome in the nucleosome fragment length span
    p1_pos = find_p1(smoothed, nuc_len_span)
    padding = vit_gen.window_padding

    # Shifted image has double the padding to allow for the placement
    # of the source data anywhere in the new image with indexing errors
    shifted_img = np.zeros((smoothed.shape[0], vit_gen.window+padding*4))

    # Place the data in the new index and crop to the appropriate window size
    new_pos = padding-p1_pos
    selected_span = new_pos, (new_pos+smoothed.shape[1])
    shifted_img[:, selected_span[0]:selected_span[1]] = smoothed
    shifted_img_crop = shifted_img[:, padding*2:padding*2+vit_gen.window]

    return shifted_img_crop, p1_pos


