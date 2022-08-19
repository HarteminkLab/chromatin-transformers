
import sys
sys.path.append('.')

import torch
import torchvision
import umap

import pandas as pd
import numpy as np
from src.timer import Timer

from src.timer import Timer
from src.plot_utils import apply_global_settings
from src.utils import mkdir_safe
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from src.gknn import entropy
from scipy.spatial.distance import euclidean


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


    def compute_umap(self, t=None):

        vit_data = self.trainer.dataloader.dataset

        tpm_lfc = vit_data.read_logfold_tpm_data(flatten=True, include_0=False)
        times = vit_data.times
        orfs = vit_data.orfs
        atts_vec = self.atts_vec

        if t is not None:
            all_indices = np.arange(len(vit_data))
            time_t_indices = all_indices[vit_data.times == t]

            if len(atts_vec) != len(time_t_indices):
                atts_vec = atts_vec[time_t_indices]

            tpm_lfc = tpm_lfc[time_t_indices]
            orfs = orfs[time_t_indices]
            times = times[time_t_indices]

        mapper = umap.UMAP().fit(atts_vec)

        tpm_all = vit_data.TPM
        plt_data = pd.DataFrame(index=orfs,
                                data={'x': mapper.embedding_[:, 0], 
                                      'y': mapper.embedding_[:, 1],
                                      'tpm_lfc': tpm_lfc,
                                      'time': times,
                                      })
        self.embeddings_df = plt_data
        self.mapper = mapper

        self.cur_atts_vec = atts_vec
        self.cur_tpm_lfc = tpm_lfc
        self.cur_orfs = orfs
        self.cur_times = times

    def plot_umap(self, t=None, selected_orfs=None, ascending=None, title="UMAP embeddings of attentions"):

        plt_data = self.embeddings_df

        if t is not None:
            plt_data = plt_data[plt_data.time == t]

        if ascending is not None:
            plt_data = plt_data.sort_values('tpm_lfc', ascending=ascending)

        plt.figure(figsize=(10, 8))

        plt.scatter(plt_data.x, plt_data.y,
                    s=10, c='#dddddd')

        plt.scatter(plt_data.x, plt_data.y, c=plt_data.tpm_lfc, edgecolor='none',
                   s=10, cmap='RdBu_r', vmin=-7, vmax=7, alpha=1.)
        plt.colorbar()

        if selected_orfs is not None:

            common_orfs = list(set(plt_data.index.values).intersection(selected_orfs))

            selected_data = plt_data.loc[common_orfs]
            plt.scatter(selected_data.x, selected_data.y, edgecolor='green', marker='D', facecolor='none',
                        s=50)

            title = f"{title}, n={len(selected_orfs)}"

        else:
            title = f"{title}, n={len(plt_data)}"

        plt.title(title, fontsize=16)

    def load_rossi(self):
        rossi = pd.read_csv('data/Rossi_Sites_All.txt', sep=' ')
        rossi_w_targets = rossi[~rossi.Target.isna()]
        tf_names = rossi_w_targets['name'].unique()
        self.rossi_w_targets = rossi_w_targets
        self.tf_names = tf_names

    def compute_rossi_dispersion(self):

        tf_names = self.tf_names
        rossi_w_targets = self.rossi_w_targets
        atts_vec = self.cur_atts_vec
        orfs_data = self.trainer.dataloader.dataset.orfs_data

        timer = Timer()
        rossi_dispersions = pd.DataFrame(index=tf_names)
        rossi_dispersions['js'] = np.nan
        rossi_dispersions['mean_euclidean'] = np.nan
        rossi_dispersions['mean_squared_euclidean'] = np.nan

        k = 1000
        i = 0

        null_distances, null_indices = compute_rand_att_vec_distances(atts_vec, k)
        null_distances_2, _ = compute_rand_att_vec_distances(atts_vec, k)
        null_entropy = compute_sampled_entropy(atts_vec, k)

        rossi_dispersions.loc['Null', 'mean_euclidean'] = null_distances.mean()
        rossi_dispersions.loc['Null', 'mean_squared_euclidean'] = (null_distances**2).mean()
        rossi_dispersions.loc['Null', 'js'] = jensenshannon(null_distances, null_distances_2)
        rossi_dispersions.loc['Null', 'n'] = k
        rossi_dispersions.loc['Null', 'entropy'] = null_entropy

        for tf in tf_names:
            selected_orfs = orfs_for_rossi_tf(orfs_data, rossi_w_targets, tf)

            if len(selected_orfs) <= 1: continue

            selected_index = index_for_selected_orfs(self.cur_orfs, selected_orfs)
            selected_atts_vec = atts_vec[selected_index]

            if selected_atts_vec.sum() == 0: continue

            selected_distances, _ = compute_rand_att_vec_distances(selected_atts_vec, k)

            rossi_dispersions.loc[tf, 'mean_euclidean'] = selected_distances.mean()
            rossi_dispersions.loc[tf, 'mean_squared_euclidean'] = (selected_distances**2).mean()
            rossi_dispersions.loc[tf, 'js'] = jensenshannon(null_distances, selected_distances)
            rossi_dispersions.loc[tf, 'n'] = len(selected_orfs)
            rossi_dispersions.loc[tf, 'entropy'] = compute_sampled_entropy(selected_atts_vec, k)

            timer.print_progress(i, len(tf_names), every=60)
            i += 1

        rossi_dispersions['entropy_difference'] = rossi_dispersions.entropy - null_entropy

        self.rossi_dispersions = rossi_dispersions
        self.null_orfs = self.cur_orfs[null_indices]
        self.null_indices = null_indices
            

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


def compute_rand_att_vec_distances(atts_vec, k):
    indices = np.random.choice(len(atts_vec), (k, 2), replace=True)

    dists = np.zeros(k)
    for i in range(len(indices)):
        dists[i] = euclidean(atts_vec[indices[i, 0]], atts_vec[indices[i, 1]])

    return dists, indices

def index_for_selected_orfs(orfs, selected):
    select_orfs = np.isin(orfs, selected)
    selected_idx = np.arange(len(orfs))[select_orfs]
    return selected_idx

def avg_dist_two_vecs(atts_vec_1, atts_vec_2, k):

    indices_1 = np.random.choice(len(atts_vec_1), k, replace=True)
    indices_2 = np.random.choice(len(atts_vec_2), k, replace=True)

    dists = np.zeros(k)
    for i in range(len(indices_1)):
        dists[i] = euclidean(atts_vec_1[indices_1[i]], atts_vec_2[indices_2[i]])

    return dists.mean()


def orfs_for_rossi_tf(orfs_data, rossi_w_targets, tf):
    targets = rossi_w_targets[rossi_w_targets['name'] == tf]
    sel_orfs = orfs_data[orfs_data['name'].isin(targets.Target.values)].index.values
    return sel_orfs


def indices_for_for_rossi_tf(orfs, orfs_data, rossi_w_targets, tf):
    sel_orfs = orfs_for_rossi_tf(orfs_data, rossi_w_targets, tf)
    return index_for_selected_orfs(orfs, sel_orfs)


def compute_sampled_entropy(atts_vec, k):
    k = min(k, len(atts_vec))
    indices = np.random.choice(len(atts_vec), k, replace=False)
    entropy_val = entropy(atts_vec[indices, :])
    return entropy_val

