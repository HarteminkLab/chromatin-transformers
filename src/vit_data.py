
import sys
sys.path.append('.')

import torch

import pandas as pd
import numpy as np

from src.utils import read_pickle
from torch.utils.data import Dataset
from src.reference_data import read_orfs_data
import torchvision.transforms as transforms
from sklearn.preprocessing import scale


class ViTData(Dataset):

    def __init__(self, all_imgs, orfs, chrs, times, TPM):

        (self.all_imgs, self.orfs, self.chrs, self.times, 
         self.TPM) = all_imgs, orfs, chrs, times, TPM

        img_transform = transforms.Normalize((0.5), (0.5), (0.5))

        self.orfs_data = read_orfs_data('data/orfs_cd_paper_dataset.csv')
        self.unscaled_TPM = TPM
        self.TPM = scale(np.log2(self.TPM+1).astype('float')).astype('float')
        self.original_imgs = self.all_imgs.copy()
        self.all_imgs = img_transform(torch.tensor(self.all_imgs))
    
    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        return self.all_imgs[idx], self.TPM[idx], self.orfs[idx], self.chrs[idx], self.times[idx]

    def unscale_log_tx(self, tx):
        mean, std = np.log2(self.unscaled_TPM+1).mean(), np.log2(self.unscaled_TPM+1).std()
        return tx*std+mean

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

    def read_log_tpm_data(self):
        tpm_df = pd.DataFrame({'orf_name': self.orfs, 'tpm': self.unscaled_TPM, 'time': self.times})
        tpm_df = tpm_df.pivot(index='orf_name', columns='time', values='tpm')
        tpm_df = np.log2(tpm_df+1)
        return tpm_df

    def index_for(self, gene_name, time):
        orf_name = self.orfs_data[(self.orfs_data['name'] == gene_name) |
                             (self.orfs_data.index == gene_name)].index.values[0]

        index = np.arange(len(self))[(self.orfs == orf_name) & 
                                       (self.times == time)][0]

        return index


def load_cd_data_12x64(replicate_mode='merge'):
    file_prefix = "vit_imgs_12x64"
    return load_cd_data(file_prefix, replicate_mode)


def load_cd_data_24x128(replicate_mode='merge'):
    file_prefix = "vit_imgs_24x128"
    return load_cd_data(file_prefix, replicate_mode)

def load_cd_data_96x512(replicate_mode='merge'):
    file_prefix = "vit_imgs_96x512"
    directory='data/vit/cd/96x512'
    return load_cd_data(file_prefix, replicate_mode, directory=directory)


def read_rna_TPM(TPM_path, orfs, times):    
    tpm_df = read_orfs_data(TPM_path, times=times)
    tpm_df = tpm_df.unstack().reset_index().rename(columns={'level_0': 'time', 0: 'TPM'})
    orfs_times = list(zip(orfs, times))
    tpm_df = tpm_df.set_index(['orf_name', 'time']).loc[orfs_times]
    TPM = tpm_df.TPM.values
    return TPM


def read_mnase_pickle(pickle_paths):
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

    # TODO: Add channel dim if it is missing
    if all_imgs.ndim == 3:
        all_imgs = all_imgs.reshape(all_imgs.shape[0], 1, all_imgs.shape[1], all_imgs.shape[2])

    return all_imgs, times, orfs, chrs, df


def load_cell_cycle_data(replicate_mode='merge'):
    
    pickle_paths_1 = (f'data/vit/cell_cycle/vit_imgs_24x128_DMAH64_MNase_rep1_0_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH66_MNase_rep1_20_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH67_MNase_rep1_30_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH68_MNase_rep1_40_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH69_MNase_rep1_50_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH70_MNase_rep1_60_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH71_MNase_rep1_70_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH72_MNase_rep1_80_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH73_MNase_rep1_90_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH74_MNase_rep1_100_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH76_MNase_rep1_120_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH77_MNase_rep1_130_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH78_MNase_rep1_140_min.pkl')

    pickle_paths_2 = (f'data/vit/cell_cycle/vit_imgs_24x128_DMAH82_MNase_rep2_0_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH83_MNase_rep2_10_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH84_MNase_rep2_20_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH85_MNase_rep2_30_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH86_MNase_rep2_40_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH87_MNase_rep2_50_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH88_MNase_rep2_60_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH89_MNase_rep2_70_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH90_MNase_rep2_80_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH91_MNase_rep2_90_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH92_MNase_rep2_100_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH94_MNase_rep2_120_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH95_MNase_rep2_130_min.pkl',
                      f'data/vit/cell_cycle/vit_imgs_24x128_DMAH96_MNase_rep2_140_min.pkl')

    rna_TPM_path = 'data/vit/cell_cycle_rna_TPM.csv'
    vit_data = load_data(pickle_paths_1, pickle_paths_2, rna_TPM_path, replicate_mode)
    return vit_data


def load_data(pickle_paths_1, pickle_paths_2, rna_TPM_path, replicate_mode):
    all_imgs_1, times, orfs, chrs, df = read_mnase_pickle(pickle_paths_1)
    all_imgs_2, _, _, _, _ = read_mnase_pickle(pickle_paths_2)

    # Merge the replicates
    if replicate_mode == 'merge':
        all_imgs = (all_imgs_1 + all_imgs_1)

    # Treat replicates as separate channels
    elif replicate_mode == 'channels':
        all_imgs = np.concatenate([all_imgs_1, all_imgs_1], axis=1)

    else:
        raise ValueError(f"Unimplemented {replicate_mode}")

    TPM = read_rna_TPM(rna_TPM_path, orfs, times)
    vit_data = ViTData(all_imgs, orfs, chrs, times, TPM)

    return vit_data


def load_cd_data(file_prefix, replicate_mode='merge', directory='data/vit/cd'):

    TPM_path = 'data/vit/cd_rna_seq_TPM.csv'

    pickle_paths_1 = (f'{directory}/{file_prefix}_DM498_MNase_rep1_0_min.pkl',
                    f'{directory}/{file_prefix}_DM499_MNase_rep1_7.5_min.pkl',
                    f'{directory}/{file_prefix}_DM500_MNase_rep1_15_min.pkl',
                    f'{directory}/{file_prefix}_DM501_MNase_rep1_30_min.pkl',
                    f'{directory}/{file_prefix}_DM502_MNase_rep1_60_min.pkl',
                    f'{directory}/{file_prefix}_DM503_MNase_rep1_120_min.pkl')

    pickle_paths_2 = (f'{directory}/{file_prefix}_DM504_MNase_rep2_0_min.pkl',
                      f'{directory}/{file_prefix}_DM505_MNase_rep2_7.5_min.pkl',
                      f'{directory}/{file_prefix}_DM506_MNase_rep2_15_min.pkl',
                      f'{directory}/{file_prefix}_DM507_MNase_rep2_30_min.pkl',
                      f'{directory}/{file_prefix}_DM508_MNase_rep2_60_min.pkl',
                      f'{directory}/{file_prefix}_DM509_MNase_rep2_120_min.pkl')

    vit_data = load_data(pickle_paths_1, pickle_paths_2, TPM_path, replicate_mode)
    return vit_data


if __name__ == '__main__':
    main()

