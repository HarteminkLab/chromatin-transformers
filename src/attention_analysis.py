
import sys
sys.path.append('.')

import torch
import torchvision

import pandas as pd
import numpy as np

from src.timer import Timer
from src.plot_utils import apply_global_settings
from src.utils import mkdir_safe
import matplotlib.pyplot as plt


class AttentionAnalysis:

    def __init__(self, trainer):
        self.trainer = trainer

    def compute_attentions(self, t):

        trainer = self.trainer
        trainer.compute_attentions(t=t)
        vit_data = trainer.dataloader.dataset

        from einops.layers.torch import Rearrange

        vectorize_att = Rearrange('b i (r) (c) -> b (i r c)')
        atts_vectorized = vectorize_att(torch.Tensor(trainer.collected_attentions)).numpy()
        self.atts_vec = atts_vectorized

        idx_t = np.arange(len(vit_data))[vit_data.times == t]
        self.orfs_t = trainer.dataloader.dataset.orfs[idx_t]


    def plot_clusters(self):

        orfs_120 = self.orfs_120
        trainer = self.trainer
        atts_120 = trainer.collected_attentions
        atts_vec_120 = self.atts_vec_120
        vit_data = trainer.dataloader.dataset
        idx_120 = np.arange(len(vit_data))[vit_data.times == 120]

        from sklearn.cluster import KMeans

        c = 16

        np.random.seed(123)
        kmeans = KMeans(n_clusters=c, random_state=1).fit(atts_vec_120)
        cd_120_cluster_labels = kmeans.labels_
        imgs_120 = vit_data.all_imgs[idx_120]
        self.cd_120_cluster_labels = cd_120_cluster_labels

        cl_df = pd.DataFrame(index=np.arange(c))

        for clus, row in cl_df.iterrows():
            
            current_idx = np.arange(len(atts_120))[cd_120_cluster_labels == clus]
            
            q0, q1, q2, q3 = 0, 0.25, 0.75, 1
            quantiles = trainer.dataloader.dataset.read_log_tpm_data().loc[orfs_120[current_idx]].quantile(
                [q0, q1, 0.5, q2, q3])

            cl_df.loc[clus, 'cluster_original'] = clus
            cl_df.loc[clus, 'q0_120'] = quantiles[120.0][q0]
            cl_df.loc[clus, 'q25_120'] = quantiles[120.0][q2]
            cl_df.loc[clus, 'med_120'] = quantiles[120.0][0.5]
            cl_df.loc[clus, 'q75_120'] = quantiles[120.0][q2]
            cl_df.loc[clus, 'q100_120'] = quantiles[120.0][q3]

        cl_df = cl_df.sort_values('med_120')
        self.cl_df = cl_df

        apply_global_settings(titlepad=10)

        plt_idx = 0
        c_max = 4

        mkdir_safe(f'{trainer.out_dir}/clusters')
        subplot_num = 0
        num_cols = 4

        for cluster in cl_df.index.values:
            
            cur_plt_idx = plt_idx % c_max

            if cur_plt_idx == 0:
                fig = plt.figure(figsize=(16, 8))
                fig.tight_layout() 
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
                subplot_num += 1

            current_idx = np.arange(len(atts_120))[cd_120_cluster_labels == cluster]
            
            plt.subplot(c_max, num_cols, (cur_plt_idx*num_cols)+1)
            plot_tpm_cluster_idx(trainer, orfs_120, current_idx)
            plt.xticks([])
                
            plt.subplot(c_max, num_cols, (cur_plt_idx*num_cols)+2)
            plt.imshow(atts_120[current_idx].mean(axis=0), origin='lower', extent=[-512, 512, 0, 3], aspect='auto')
            plt.axvline(0, color='white', lw=1, linestyle='dotted')
            plt.xticks([])
            plt.yticks([])
            plt.title(f"Cluster {plt_idx+1}, n={len(current_idx)}")
            
            plt.subplot(c_max, num_cols, (cur_plt_idx*num_cols)+3)
            imgs_120 = vit_data.all_imgs[idx_120]
            imgs_120[current_idx].mean(axis=0).shape
            plt.imshow(imgs_120[current_idx].mean(axis=0)[0], origin='lower', extent=[-8, 8, 0, 3], 
                       cmap='magma_r', aspect='auto')
            plt.axvline(0, color='gray', lw=1, linestyle='dotted')
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(c_max, num_cols, (cur_plt_idx*num_cols)+3)
            imgs_120 = vit_data.all_imgs[idx_120]
            imgs_120[current_idx].mean(axis=0).shape
            plt.imshow(imgs_120[current_idx].mean(axis=0)[0], origin='lower', extent=[-8, 8, 0, 3], 
                       cmap='magma_r', aspect='auto')
            plt.axvline(0, color='gray', lw=1, linestyle='dotted')
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(c_max, num_cols, (cur_plt_idx*num_cols)+4)
            imgs_120 = vit_data.all_imgs[idx_120]
            imgs_120[current_idx].mean(axis=0).shape
            plt.imshow(imgs_120[current_idx].mean(axis=0)[0], origin='lower', extent=[-8, 8, 0, 3], 
                       cmap='magma_r', aspect='auto')
            
            from matplotlib.colors import LinearSegmentedColormap
            cm_data = [[0, 0, 0, 1], [0, 0, 0, 0]]
            color_map = LinearSegmentedColormap.from_list('overlay', cm_data)
            plt.imshow(atts_120[current_idx].mean(axis=0), origin='lower', 
                       cmap=color_map, extent=[-8, 8, 0, 3], aspect='auto',
                      vmin=0, vmax=1, alpha=0.75)
            
            
            plt.axvline(0, color='gray', lw=1, linestyle='dotted')
            plt.xticks([])
            plt.yticks([])

            if cur_plt_idx == (c_max-1):
               plt.savefig(f'{trainer.out_dir}/clusters/{c}_{subplot_num}.png', dpi=200)

            plt_idx += 1

    def plot_cluster_examples(self, cluster):
        
        cluster = cluster-1

        orfs_120 = self.orfs_120
        trainer = self.trainer
        atts_120 = trainer.collected_attentions
        atts_vec_120 = self.atts_vec_120
        vit_data = trainer.dataloader.dataset
        idx_120 = np.arange(len(vit_data))[vit_data.times == 120]

        imgs_120 = vit_data.all_imgs[idx_120]
        cd_120_cluster_labels = self.cd_120_cluster_labels

        np.random.seed(123)
        num_examples = 9
        current_idx = np.arange(len(atts_120))[cd_120_cluster_labels == cluster]
        random_idxs = np.random.choice(current_idx, num_examples, replace=False)

        fig = plt.figure(figsize=(6, 2))
        imgs_120 = vit_data.all_imgs[idx_120]
        imgs_120[current_idx].mean(axis=0).shape

        for i in range(num_examples):
            index = random_idxs[i]
            plt.subplot(3, 3, i+1)
            plt.imshow(imgs_120[index, 0], origin='lower', extent=[-8, 8, 0, 3], 
                       cmap='magma_r', aspect='auto')
            plt.xticks([])
            plt.yticks([])
            plt.axvline(0, color='gray', lw=1, linestyle='dotted')


def plot_tpm_cluster_idx(trainer, orfs_120, current_idx):
    
    q0, q1, q2, q3 = 0.0, 0.25, 0.75, 1.
    quantiles = trainer.dataloader.dataset.read_log_tpm_data().loc[orfs_120[current_idx]].quantile(
        [q0, q1, 0.5, q2, q3])

    meds = quantiles.loc[0.5]
    q0s = quantiles.loc[q0]
    q1s = quantiles.loc[q1]
    q2s = quantiles.loc[q2]
    q3s = quantiles.loc[q3]
    
    x = range(len(quantiles.columns))
    
    plt.plot(x, meds.values, '-', c='purple')
    plt.plot(x, meds.values, 'D', c='black', zorder=5)
    
    cspan = 4, 7
    
    val = meds[120]
    
    plt.fill_between(x, q1s, q2s, color=plt.get_cmap('Purples')
                     ((val-cspan[0])/(cspan[1]-cspan[0])), 
                     alpha=1., zorder=2)
    
    plt.fill_between(x, q0s, q3s, color=plt.get_cmap('Purples')
                     ((val-cspan[0])/(cspan[1]-cspan[0])), 
                     alpha=0.5, zorder=2)

    ax = plt.gca()
    plt.xticks(x, quantiles.columns)
    plt.yticks(np.arange(0, 20, 8))
    ax.set_yticks(np.arange(0, 20, 4), minor=True)
    plt.ylim(-1, 16)
    plt.xlim(0, 5)
    plt.ylabel("$\\log$ TPM")


def plot_umap(vit_data, mapper, clus_labels, log_tpm, selected, hide_ticks=False):

    idx_120 = np.arange(len(vit_data))[vit_data.times == 120]
    orfs_120 = vit_data.orfs[idx_120]

    contain = np.isin(orfs_120, selected)
    selected_idx = np.arange(len(orfs_120))[contain]

    mapper_x = mapper.embedding_[:, 0]
    mapper_y = mapper.embedding_[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(mapper_x, mapper_y, s=15, facecolor='none', 
                c='#dddddd', zorder=1)
    plt.scatter(mapper_x, mapper_y, s=10, edgecolor='none', 
                c=clus_labels, cmap='Spectral', alpha=1., zorder=2)
    plt.scatter(mapper_x[contain], mapper_y[contain], s=40, marker='D', 
                facecolor='none', edgecolor='black', alpha=1., zorder=3)
    plt.xticks([])
    plt.yticks([])
    plt.title("UMAP embeddings of t=120 attentions colored by cluster labels (k=16)")

    bbox = plt.gca().get_window_extent()
    cax = plt.axes([0.95, 0.125, 0.015, 0.755])
    cax.yaxis.tick_right()

    cax.imshow(np.arange(16).reshape(16, 1), extent=[0, 1, 0, 16], zorder=100, cmap='Spectral_r', aspect='auto')
    cax.set_yticks(np.arange(16)+0.5)
    cax.set_yticklabels(np.arange(16)+1)
    cax.set_xticks([])

    plt.figure(figsize=(7.5, 6))
    plt.title("UMAP embeddings of t=120 attentions colored by log2 Transcript level")
    plt.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1], s=15, facecolor='none', 
                c='#dddddd', zorder=1)
    plot_data = pd.DataFrame(data={'x':mapper.embedding_[:, 0], 
                       'y':mapper.embedding_[:, 1],
                       'tpm':log_tpm})
    plot_data = plot_data.sort_values('tpm')
    plt.scatter(plot_data.x, plot_data.y, s=10, edgecolor='none', 
                c=plot_data.tpm, cmap='viridis', vmin=5, vmax=12, alpha=1., zorder=2)
    plt.colorbar()
    plt.scatter(mapper_x[contain], mapper_y[contain], s=60, marker='D', 
                    facecolor='none', edgecolor='red', alpha=0.75, zorder=3)

    if hide_ticks:
        plt.xticks([])
        plt.yticks([])

    for i in range(0, 14, 1):
        plt.axvline(i, c='#ddd', lw=1, zorder=1, linestyle='dotted')
        plt.axhline(i, c='#ddd', lw=1, zorder=1, linestyle='dotted')
