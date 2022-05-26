
import sys
sys.path.append('.')

import pandas as pd
import numpy as np

from src.utils import read_pickle
from torch.utils.data import Dataset
from src.reference_data import read_orfs_data
import torchvision.transforms as transforms
import torch
from sklearn.preprocessing import scale


class ViTData(Dataset):

    def __init__(self, all_imgs, orfs, chrs, times, TPM):
        (self.all_imgs, self.orfs, self.chrs, self.times, 
         self.TPM) = all_imgs, orfs, chrs, times, TPM

        img_transform = transforms.Normalize((0.5), (0.5), (0.5))

        self.unscaled_TPM = TPM
        self.TPM = scale(np.log2(self.TPM+1).astype('float')).astype('float')
        self.original_imgs = self.all_imgs.copy()
        self.all_imgs = img_transform(torch.tensor(self.all_imgs))
    
    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        return self.all_imgs[idx], self.TPM[idx], self.orfs[idx], self.chrs[idx], self.times[idx]

    def create_tpm_df(self):
        data_df = pd.DataFrame({
            'orf_name': self.dataloader.dataset.orfs,
            'time': self.dataloader.dataset.times,
            'chr': self.dataloader.dataset.chrs,
            'TPM': self.dataloader.dataset.unscaled_TPM,
        })
        orfs = read_orfs_data('data/orfs_cd_paper_dataset.csv')

        tpm_data = data_df.pivot(index='orf_name', values='TPM', columns='time')
        tpm_0 = tpm_data[0.0].copy()
        for time in tpm_data.columns:
            tpm_data[time] = np.log2((tpm_data[time]+1) / (tpm_0+1))

        tpm_data = tpm_data.sort_values(120.0, ascending=False).join(orfs[['name']])
        return tpm_data


def load_cd_data():

    pdata = read_pickle('data/vit/vit_data_gene_TSS_mnase_32_128.pkl')

    (all_imgs, TPM, 
     chrs, orfs,
     times) = pdata

    vit_data = ViTData(all_imgs, orfs, chrs, times, TPM)

    return vit_data


if __name__ == '__main__':
    main()

