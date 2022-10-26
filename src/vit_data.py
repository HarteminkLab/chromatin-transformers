
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

    def __init__(self, all_imgs, orfs, chrs, times, TPM, channel_1_time, predict_tpm):

        (self.all_imgs, self.orfs, self.chrs, self.times, 
         self.TPM) = all_imgs, orfs, chrs, times, TPM

        self.original_imgs = self.all_imgs.copy()
        self.original_chrs = self.chrs
        self.original_TPM = self.TPM
        self.original_orfs = self.orfs
        self.original_times = self.times
        self.unscaled_TPM = self.TPM
        self.predict_tpm = predict_tpm
        self.all_imgs_untransformed = self.all_imgs.copy()
        self.img_transform = transforms.Normalize((0.5), (0.5), (0.5))
        self.orfs_data = read_orfs_data('data/orfs_cd_paper_dataset.csv')
        self.init_transforms()

    def init_transforms(self):

        # Predict absolute expression level
        if predict_tpm == 'absolute':
            self.TPM = scale(np.log2(self.TPM+1).astype('float')).astype('float')
        # Predict logfold change in expression
        elif predict_tpm == 'logfold':
            logfold_TPM = self.read_logfold_tpm_data(flatten=True)
            self.TPM = scale(logfold_TPM.astype('float')).astype('float')
        else:
            raise ValueError(f"Unsupported TPM prediction {predict_tpm}")

        self.all_imgs = img_transform(torch.tensor(self.all_imgs))

        if channel_1_time is not None:
            (all_imgs, chrs, TPM, orfs, times) = \
                self.transform_data_for_channel(channel_1_time, self.all_imgs)

            (self.all_imgs_untransformed_channeled, _, _,_, _) = \
                self.transform_data_for_channel(channel_1_time, self.all_imgs_untransformed)

            self.all_imgs = all_imgs
            self.chrs = chrs
            self.TPM = TPM
            self.orfs = orfs
            self.times = times

    def transform_data_for_channel(self, channel_1_time, all_imgs):

        n = len(self.all_imgs)
        all_indices = np.arange(n)

        if channel_1_time == None:
            images_1 = None
            images_t = all_imgs
            time_t_indices = all_indices
        # Use one of the other time points as a channel
        else:
            time_1_indices = all_indices[self.times == channel_1_time]
            time_t_indices = all_indices[self.times != channel_1_time]
            images_1 = all_imgs[time_1_indices]
            images_t = all_imgs[time_t_indices]

        if images_1 is not None:
            # Duplicate channel 1 images so it matches remaining dataset
            repeat = images_t.shape[0]//images_1.shape[0]
            images_1 = np.tile(images_1, (repeat, 1, 1, 1))
            # Concatenate images to two channels, 
            all_imgs = np.concatenate([images_1, images_t], axis=1)
        else:
            all_imgs = images_t

        # reselect remaining data to match removing channel 1 data
        chrs = self.chrs[time_t_indices]
        TPM = self.TPM[time_t_indices]
        orfs = self.orfs[time_t_indices]
        times = self.times[time_t_indices]

        return all_imgs, chrs, TPM, orfs, times

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        return self.all_imgs[idx], self.TPM[idx], self.orfs[idx], self.chrs[idx], self.times[idx]

    def unscale_log_tx(self, tx):
        mean, std = np.log2(self.unscaled_TPM+1).mean(), np.log2(self.unscaled_TPM+1).std()
        return tx*std+mean

    def read_lfc_mean(self):
        tpm_data = self.read_tpm_data()
        mean_tpm = tpm_data.mean(axis=1)
        lfc_mean_tpm = tpm_data.copy()

        for t in tpm_data.columns:
            lfc_mean_tpm[t] = np.log2((tpm_data[t] + 1) / (mean_tpm + 1))
        return lfc_mean_tpm

    def read_tpm_data(self):
        tpm_df = pd.DataFrame({'orf_name': self.original_orfs, 'tpm': self.unscaled_TPM, 'time': self.original_times})
        tpm_df = tpm_df.pivot(index='orf_name', columns='time', values='tpm')
        return tpm_df

    def read_log_tpm_data(self, flatten=False):
        tpm_df = self.read_tpm_data()
        tpm_df = np.log2(tpm_df.loc[self.orfs_data.index.values]+1)
        if flatten:
            return tpm_df.values.reshape(-1, order='F')
        return tpm_df

    def read_logfold_tpm_data(self, flatten=False, include_0=True):
        tpm_df = self.read_tpm_data()
        tpm_0 = tpm_df[0.0].copy()
        for time in tpm_df.columns:
            tpm_df[time] = np.log2((tpm_df[time]+1) / (tpm_0+1))
        tpm_df = tpm_df.loc[self.orfs_data.index.values]

        if flatten:
            
            if not include_0:
                tpm_df = tpm_df[tpm_df.columns[1:]]

            tpm = tpm_df.values.reshape(-1, order='F')

            return tpm

        return tpm_df

    def index_for(self, gene_name, time):
        orf_name = self.orfs_data[(self.orfs_data['name'] == gene_name) |
                                  (self.orfs_data.index == gene_name)].index.values[0]

        index = np.arange(len(self))[(self.orfs == orf_name) & 
                                       (self.times == time)][0]
        return index

    def plot_genes_time(self, gene_names, time):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 4))

        for i in range(len(gene_names)):
            gene_name = gene_names[i]
            plt.subplot(4, 4, i+1)
            self.plot_gene_time(gene_name, time, fig)
            plt.yticks([])

    def plot_gene_time(self, gene_name, time, fig=None):
        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.figure(figsize=(5.5, 1.5))

        idx = self.index_for(gene_name, time)
        dat = self.all_imgs[idx]

        plt.imshow(dat[1], cmap='magma_r', vmin=-1, vmax=-0.5, origin='lower', 
            extent=[-512, 512, 0, 225], aspect='auto', interpolation='none')
        plt.xticks([])
        plt.ylim(20, 225)
        plt.axvline(0, c='blue', lw=1, linestyle='solid')

    def indices_for(self, sel_orfs, time):

        indices = np.arange(len(self))
        orf_indices = indices[np.isin(self.orfs, sel_orfs)]
        time_indices = indices[self.times == time]

        sel_indices = np.array(sorted(list(set(orf_indices).intersection(set(time_indices)))))
        return sel_indices


def load_cd_data_12x64(replicate_mode, channel_1_time, predict_tpm):
    file_prefix = "vit_imgs_12x64"
    return load_cd_data(file_prefix, replicate_mode, channel_1_time, predict_tpm)


def load_cd_data_24x128(replicate_mode, channel_1_time, predict_tpm):
    file_prefix = "vit_imgs_24x128"
    return load_cd_data(file_prefix, replicate_mode, channel_1_time, predict_tpm, 
        directory='data/vit/cd/24x128')


def load_cd_data_24x128_p1(replicate_mode, channel_1_time, predict_tpm):
    file_prefix = "vit_imgs_24x128"
    return load_cd_data(file_prefix, replicate_mode, channel_1_time, predict_tpm, 
        directory='data/vit/cd/24x128_p1')


def load_cd_data_96x512(replicate_mode, channel_1_time, predict_tpm):
    file_prefix = "vit_imgs_96x512"
    directory='data/vit/cd/96x512'
    return load_cd_data(file_prefix, replicate_mode, channel_1_time, predict_tpm, directory=directory)


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


def load_cell_cycle_data(replicate_mode, channel_1_time, predict_tpm, 
    init_class=ViTData, debug_n=None):

    data_dir = 'data/vit/cell_cycle'
    file_prefix = 'vit_imgs_24x128'
    
    pickle_paths_1 = (f'{data_dir}/{file_prefix}_DMAH64_MNase_rep1_0_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH66_MNase_rep1_20_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH67_MNase_rep1_30_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH68_MNase_rep1_40_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH69_MNase_rep1_50_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH70_MNase_rep1_60_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH71_MNase_rep1_70_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH72_MNase_rep1_80_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH73_MNase_rep1_90_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH74_MNase_rep1_100_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH76_MNase_rep1_120_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH77_MNase_rep1_130_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH78_MNase_rep1_140_min.pkl')

    pickle_paths_2 = (f'{data_dir}/{file_prefix}_DMAH82_MNase_rep2_0_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH84_MNase_rep2_20_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH85_MNase_rep2_30_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH86_MNase_rep2_40_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH87_MNase_rep2_50_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH88_MNase_rep2_60_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH89_MNase_rep2_70_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH90_MNase_rep2_80_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH91_MNase_rep2_90_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH92_MNase_rep2_100_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH94_MNase_rep2_120_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH95_MNase_rep2_130_min.pkl',
                      f'{data_dir}/{file_prefix}_DMAH96_MNase_rep2_140_min.pkl')

    TPM_path = 'data/vit/cell_cycle_rna_TPM.csv'
    vit_data = load_data(pickle_paths_1, pickle_paths_2, TPM_path, replicate_mode, 
        channel_1_time, predict_tpm, init_class=init_class, debug_n=debug_n)

    return vit_data


def load_data(pickle_paths_1, pickle_paths_2, rna_TPM_path, replicate_mode, 
              channel_1_time, predict_tpm, init_class=ViTData, debug_n=None):
    all_imgs_1, times, orfs, chrs, df = read_mnase_pickle(pickle_paths_1)
    all_imgs_2, _, _, _, _ = read_mnase_pickle(pickle_paths_2)

    # Merge the replicates
    if replicate_mode == 'merge':
        all_imgs = (all_imgs_1 + all_imgs_2)

    # Treat replicates as separate channels
    elif replicate_mode == 'channels':
        all_imgs = np.concatenate([all_imgs_1, all_imgs_1], axis=1)

    else:
        raise ValueError(f"Unimplemented {replicate_mode}")

    TPM = read_rna_TPM(rna_TPM_path, orfs, times)
    vit_data = init_class(all_imgs, orfs, chrs, times, TPM, channel_1_time, predict_tpm, debug_n=debug_n)

    return vit_data


def load_cd_data(file_prefix, replicate_mode, channel_1_time, predict_tpm, directory='data/vit/cd'):

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

    vit_data = load_data(pickle_paths_1, pickle_paths_2, TPM_path, replicate_mode, channel_1_time, predict_tpm)
    return vit_data


if __name__ == '__main__':
    main()

