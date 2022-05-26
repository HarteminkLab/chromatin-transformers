
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


def load_cd_data_12x64():
    file_prefix = "vit_imgs_12x64"
    return load_cd_data(file_prefix)

def load_cd_data_24x128():
    file_prefix = "vit_imgs_24x128"
    return load_cd_data(file_prefix)

def load_cd_data(file_prefix):

    pickle_paths = (f'data/vit/{file_prefix}_DM498_MNase_rep1_0_min.pkl',
                    f'data/vit/{file_prefix}_DM499_MNase_rep1_7.5_min.pkl',
                    f'data/vit/{file_prefix}_DM500_MNase_rep1_15_min.pkl',
                    f'data/vit/{file_prefix}_DM501_MNase_rep1_30_min.pkl',
                    f'data/vit/{file_prefix}_DM502_MNase_rep1_60_min.pkl',
                    f'data/vit/{file_prefix}_DM503_MNase_rep1_120_min.pkl'
                    )

    TPM_path = 'data/vit/cd_rna_seq_TPM.csv'

    i = 0
    df = pd.DataFrame()
    times = np.array([])
    orfs = np.array([])
    chrs = np.array([])
    all_imgs = None

    for path in pickle_paths:        
        filesplit = path.split('/')[-1].split('_')[2:]
        dm, rep, time = filesplit[0], filesplit[2], filesplit[-2]
        desc, imgs = read_pickle(path)

        if all_imgs is None: all_imgs = imgs
        else:
            all_imgs = np.concatenate([all_imgs, imgs])

        times = np.append(times, np.repeat(float(time), len(imgs)))
        orfs = np.append(orfs, np.array(desc['orfs']))
        chrs = np.append(chrs, np.array(desc['chrs']))

        df.loc[i, 'DM'] = dm
        df.loc[i, 'replicate'] = rep
        df.loc[i, 'time'] = float(time)
        df.loc[i, 'path'] = path
        i += 1
    
    tpm_df = read_orfs_data(TPM_path)
    tpm_df = tpm_df.unstack().reset_index().rename(columns={'level_0': 'time', 0: 'TPM'})
    orfs_times = list(zip(orfs, times))
    tpm_df = tpm_df.set_index(['orf_name', 'time']).loc[orfs_times]
    TPM = tpm_df.TPM.values

    vit_data = ViTData(all_imgs, orfs, chrs, times, TPM)

    return vit_data


if __name__ == '__main__':
    main()

