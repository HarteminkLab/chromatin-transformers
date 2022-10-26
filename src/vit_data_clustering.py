
import sys
sys.path.append('.')

import torch

import pandas as pd
import numpy as np
import torchvision.transforms as transforms

from src.utils import read_pickle
from torch.utils.data import Dataset
from src.reference_data import read_orfs_data
from sklearn.preprocessing import scale
from src.vit_data import ViTData


class ViTDataDeepClustering(ViTData):

    def __init__(self, all_imgs, orfs, chrs, times, TPM, channel_1_time, predict_tpm, debug_n=None):

        if debug_n is not None:
            
            indices = np.arange(len(all_imgs))[times == 20]

            all_imgs = all_imgs[indices]
            orfs = orfs[indices]
            chrs = chrs[indices]
            times = times[indices]
            TPM = TPM[indices]

        ViTData.__init__(self, all_imgs, orfs, chrs, times, TPM, channel_1_time, predict_tpm)
        self.pseudo_labels = np.zeros(len(all_imgs))
     
    def __getitem__(self, idx):
        return (idx, self.all_imgs[idx], self.pseudo_labels[idx])

    def init_transforms(self):
        self.all_imgs = self.img_transform(torch.tensor(self.all_imgs)).detach().numpy()

    def get_label_counts(self):
        labels = self.pseudo_labels
        label_counts = np.array(np.unique(labels, return_counts=True)).T

        n = len(labels)
        summary_str = []
        summary_counts = {}

        for i in range(len(label_counts)):
            label = int(label_counts[i, 0])
            count = int(label_counts[i, 1])
            summary_counts[label] = count
            summary_str.append(f"{label}: {count}/{n} ({count/n*100:.0f}%)")
            
        return summary_counts, ", ".join(summary_str)
