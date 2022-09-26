
import sys
sys.path.append('.')

import os
import torch
import torchvision
import umap
import pacmap
import cv2

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from src.timer import Timer
from src.plot_utils import apply_global_settings
from src.utils import mkdir_safe, read_pickle, write_pickle, print_fl
from src.gknn import entropy
from scipy.spatial.distance import jensenshannon
from scipy.spatial.distance import euclidean
from einops.layers.torch import Rearrange


class AttentionAnalysis:

    def __init__(self, trainer):
        self.trainer = trainer
        self.vectorize_grid = Rearrange('b i (r) (c) -> b (i r c)')


    def compute_attentions(self, t, lazy_load=True):
        trainer = self.trainer

        write_path = f"{self.trainer.out_dir}/collected_attentions_all.pkl" 

        if not lazy_load or not os.path.exists(write_path):
            trainer.compute_attentions(t=t)
            write_pickle(self.trainer.collected_attentions, write_path)
        else:
            print_fl(f"Reading existing collected attentions {write_path}")
            trainer.collected_attentions = read_pickle(f"{self.trainer.out_dir}/collected_attentions_all.pkl")

        self.compute_atts_vector(t)


    def compute_atts_vector(self, t):

        trainer = self.trainer
        vit_data = trainer.dataloader.dataset

        atts_vectorized = self.vectorize_grid(torch.Tensor(trainer.collected_attentions)).numpy()
        self.atts_vec = atts_vectorized

        idx_t = np.arange(len(vit_data))[vit_data.times == t]
        self.orfs_t = trainer.dataloader.dataset.orfs[idx_t]


    def compute_embeddings(self, t=None, ts=None, vector=None, combine_ts='narrow'):

        vit_data = self.trainer.dataloader.dataset

        tpm_lfc = vit_data.read_logfold_tpm_data(flatten=True, include_0=False)
        times = vit_data.times
        orfs = vit_data.orfs

        if t is not None:
            ts = [t]

        if vector is None:
            vector = self.atts_vec

        if ts is not None:

            if combine_ts == 'wide':
                all_indices = np.arange(len(vit_data))

                atts_vec = self.atts_vec

                t_vecs = []
                select_t = False
                for t in ts:
                    cur_sel = (vit_data.times == t)
                    cur_vec = atts_vec[cur_sel]
                    select_t = select_t | cur_sel
                    t_vecs.append(cur_vec)

                vector = np.concatenate(t_vecs, axis=1)

                time_t_indices = all_indices[select_t]

                tpm_lfc = tpm_lfc[time_t_indices]
                orfs = orfs[np.arange(len(vector))]
                times = times[time_t_indices]
            elif combine_ts == 'narrow':
                all_indices = np.arange(len(vit_data))

                atts_vec = vector

                t_vecs = []
                select_t = False
                for t in ts:
                    cur_sel = (vit_data.times == t)
                    cur_vec = atts_vec[cur_sel]
                    select_t = select_t | cur_sel
                    t_vecs.append(cur_vec)

                vector = np.concatenate(t_vecs, axis=0)

                time_t_indices = all_indices[select_t]

                tpm_lfc = tpm_lfc[time_t_indices]
                orfs = orfs[np.arange(len(vector))]
                times = times[time_t_indices]

        embedding = pacmap.PaCMAP(n_components=2, random_state=123)
        pacmap_embeddings = embedding.fit_transform(vector)

        tpm_all = vit_data.TPM
        plt_data = pd.DataFrame(index=orfs,
                                data={'x': pacmap_embeddings[:, 0], 
                                      'y': pacmap_embeddings[:, 1],
                                      'time': times,
                                      'tpm': tpm_lfc,
                                      })
        self.pacmap_embeddings_df = plt_data
        self.cur_vector = vector
        self.cur_tpm_lfc = tpm_lfc
        self.cur_orfs = orfs
        self.cur_times = times


    def plot_embeddings(self, t=None, selected_orfs=None, ascending=None,
        title="Embeddings of attentions", xs=None, ys=None, which='pacmap', fig=None,
        xlim=None, ylim=None):

        from src.plot_utils import plot_rect

        if which == 'pacmap':
            plt_data = self.pacmap_embeddings_df
        elif which == 'umap':
            plt_data = self.umap_embeddings_df

        if t is not None:
            plt_data = plt_data[plt_data.time == t]

        if ascending is not None:
            plt_data = plt_data.sort_values('tpm_lfc', ascending=ascending)
        self.plt_data = plt_data

        if fig is None:
            fig = plt.figure(figsize=(3, 3))

        plt.scatter(plt_data.x, plt_data.y,
                    s=10, c='#dddddd')

        plt.scatter(plt_data.x, plt_data.y, edgecolor='none', c=plt_data.tpm,
                   s=10, cmap='RdBu_r', vmin=-4, vmax=4, alpha=1.)
        ret = None
        if selected_orfs is not None:

            times = plt_data.time.unique()

            common_orfs = list(set(plt_data.index.values).intersection(selected_orfs))
            for orf in common_orfs:
                selected_data = plt_data.loc[orf].copy()
                selected_data['color'] = 'purple'
                colors = plt.get_cmap('magma_r')

                for i in range(1, len(times)):
                    prev_i = i-1
                    prev_pt = selected_data[selected_data.time == times[prev_i]].loc[orf]
                    pt = selected_data[selected_data.time == times[i]].loc[orf]
                    plt.plot([prev_pt.x, pt.x], [prev_pt.y, pt.y], c=colors(i/len(times)))

                plt.scatter(selected_data.x, selected_data.y,
                    edgecolor=selected_data.color, marker='D', facecolor='none',
                            s=50)

            title = f"{title}"#, n={len(selected_orfs)}"
            ret = selected_data

        ax = plt.gca()

        if xs is not None and ys is not None:
            plot_rect(ax, xs[0], ys[0], xs[1]-xs[0], ys[1]-ys[0], lw=1.0, edgecolor='red', facecolor='none')

            sel_data = plt_data[(plt_data.x >= xs[0]) & (plt_data.x < xs[1]) & 
                                (plt_data.y < ys[1]) & (plt_data.y > ys[0])]
            title = f"{title}, n={len(sel_data)}"
            ret = sel_data

        plt.title(title, fontsize=16)


        if xlim is not None:
            plt.xlim(xlim)

        if ylim is not None:
            plt.ylim(ylim)

        return ret


    def load_rossi(self):
        rossi = pd.read_csv('data/Rossi_Sites_All.txt', sep=' ')
        rossi_w_targets = rossi[~rossi.Target.isna()]
        tf_names = rossi_w_targets['name'].unique()
        self.rossi_w_targets = rossi_w_targets
        self.tf_names = tf_names

        orfs_data = self.trainer.dataloader.dataset.orfs_data
        tf_target_orfs = pd.DataFrame(index=tf_names)
        tf_target_orfs['orfs'] = None

        for tf in tf_names:
            selected_orfs = orfs_for_rossi_tf(orfs_data, rossi_w_targets, tf)
            tf_target_orfs.loc[tf, 'orfs'] = selected_orfs

        self.rossi_tf_target_orfs = tf_target_orfs


    def load_go_terms(self):
        from src.reference_data import read_sgd_orfs, go_get_term_names
        from goatools.obo_parser import GODag        

        go_obo_path = 'data/go.obo'
        obodag = GODag(go_obo_path)

        sgd_orfs = read_sgd_orfs()
        ontology_counts = {}

        for i in range(len(sgd_orfs)):
            orf_name = sgd_orfs.index.values[i]
            row = sgd_orfs.loc[orf_name]
            cur_ontologies = row['ontology'].split(',')
            
            for ont in cur_ontologies:
                if ont not in ontology_counts.keys():
                    ontology_counts[ont] = np.array([orf_name])
                else: ontology_counts[ont] = np.append(ontology_counts[ont], orf_name)

        go_orfs = pd.DataFrame(index=ontology_counts.keys(), data={'orfs': ontology_counts.values()})
        term_names = go_get_term_names(obodag, go_orfs.index.values)
        go_orfs['description'] = term_names

        self.go_orfs = go_orfs


    def compute_rossi_dispersion(self):
        rossi_dispersions = self.compute_group_orfs_dispersion(self.rossi_tf_target_orfs)
        self.rossi_dispersions = rossi_dispersions


    def compute_group_orfs_dispersion(self, group_orfs):

        group_names = group_orfs.index.values
        rossi_w_targets = self.rossi_w_targets
        atts_vec = self.cur_vector
        orfs_data = self.trainer.dataloader.dataset.orfs_data

        timer = Timer()
        dispersions_df = pd.DataFrame(index=group_names)
        dispersions_df['js'] = np.nan
        dispersions_df['mean_euclidean'] = np.nan
        dispersions_df['mean_squared_euclidean'] = np.nan

        k = 1000
        i = 0

        null_distances, null_indices = compute_rand_att_vec_distances(atts_vec, k)
        null_distances_2, _ = compute_rand_att_vec_distances(atts_vec, k)
        null_entropy = compute_sampled_entropy(atts_vec, k)

        dispersions_df.loc['Null', 'mean_euclidean'] = null_distances.mean()
        dispersions_df.loc['Null', 'mean_squared_euclidean'] = (null_distances**2).mean()
        dispersions_df.loc['Null', 'js'] = jensenshannon(null_distances, null_distances_2)
        dispersions_df.loc['Null', 'n'] = k
        dispersions_df.loc['Null', 'entropy'] = null_entropy

        for group_name in group_names:
            selected_orfs = group_orfs.loc[group_name].orfs

            if len(selected_orfs) <= 1: continue

            selected_index = index_for_selected_orfs(self.cur_orfs, selected_orfs)
            selected_atts_vec = atts_vec[selected_index]

            if selected_atts_vec.sum() == 0: continue

            selected_distances, _ = compute_rand_att_vec_distances(selected_atts_vec, k)

            dispersions_df.loc[group_name, 'mean_euclidean'] = selected_distances.mean()
            dispersions_df.loc[group_name, 'mean_squared_euclidean'] = (selected_distances**2).mean()
            dispersions_df.loc[group_name, 'js'] = jensenshannon(null_distances, selected_distances)
            dispersions_df.loc[group_name, 'n'] = len(selected_orfs)
            dispersions_df.loc[group_name, 'entropy'] = compute_sampled_entropy(selected_atts_vec, k)

            timer.print_progress(i, len(group_names), every=60)
            i += 1

        dispersions_df['entropy_difference'] = dispersions_df.entropy - null_entropy
        return dispersions_df
            

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


    def plot_selected_examples(self, sel_data):

        vit_data = self.trainer.dataloader.dataset
        sel_indices = vit_data.indices_for(sel_data.index.values, 100)
        sel_atts = self.trainer.collected_attentions[sel_indices]
        sel_indices = sel_indices[0:16]

        fig = plt.figure(figsize=(6, 1.5))
        plt.imshow(sel_atts.mean(axis=0)[1], origin='lower', extent=[-512, 512, 0, 200])
        plt.yticks([])
        plt.xticks([])
        plt.axvline(0)

        fig = plt.figure(figsize=(12, 4))
        for i in range(len(sel_indices)):
            plt.subplot(4, 4, i+1)
            plt.imshow(sel_atts[i, 1], origin='lower', extent=[-512, 512, 0, 200])
            plt.yticks([])
            plt.xticks([])
            plt.axvline(0)

        vit_data.plot_genes_time(vit_data.orfs[sel_indices], 100)

    def resize_imgs(self):

        trainer = self.trainer
        vit_data = trainer.dataloader.dataset

        timer = Timer()
        att_size = trainer.collected_attentions.shape[-2], trainer.collected_attentions.shape[-1]

        imgs_resized = np.zeros_like(trainer.collected_attentions)
        for i in range(len(vit_data.all_imgs)):
            cur_img = vit_data.all_imgs[i]
            for channel in range(cur_img.shape[0]):
                img_resize = cv2.resize(cur_img[channel], (att_size[1], att_size[0]))
                imgs_resized[i, channel] = img_resize
            timer.print_progress(i, len(vit_data.all_imgs), every=20000)

        self.imgs_resized = imgs_resized
        self.combined_att_imgs = trainer.collected_attentions * (imgs_resized + 1.0)
        
        write_pickle(imgs_resized, f"{trainer.out_dir}/imgs_resized.pkl")
        write_pickle(self.combined_att_imgs, f"{trainer.out_dir}/combined_att_imgs.pkl")

        self.imgs_resized_vec = self.vectorize_grid(torch.Tensor(self.imgs_resized)).numpy()
        self.combined_att_imgs_vec = self.vectorize_grid(torch.Tensor(self.combined_att_imgs)).numpy()


    def plot_att_gene_3d(self, ret, times, gene_name, subplot=111, fig=None, xlim=None, ylim=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        plt.rcParams['grid.color'] = "#eee"

        if fig is None:
            fig = plt.figure(figsize=(6, 6))

        ax = fig.add_subplot(subplot, projection='3d')
        ax.view_init(elev=30, azim=45)

        xs = [0, 1]
        ys = [0, 1]

        cur_dat = self.pacmap_embeddings_df
        dat = cur_dat

        ax.scatter(dat.x, dat.y, zs=dat.time, zdir='z', s=1, c=dat.tpm, cmap='RdBu_r', 
            vmin=-5, vmax=5, alpha=0.01)

        cmap = plt.get_cmap('magma_r')
        colors = [cmap(i/len(times)) for i in range(len(times))]

        ax.scatter(ret.x, ret.y, zs=ret.time, zdir='z', s=20, c=colors)

        for i in range(1, len(times)):
            prev_i = i-1
            prev_pt = ret[ret.time == times[prev_i]].iloc[0]
            pt = ret[ret.time == times[i]].iloc[0]
            plt.plot([prev_pt.x, pt.x], [prev_pt.y, pt.y], zs=[times[i-1], times[i]], c=colors[i])


        if xlim is not None:
            plt.xlim(xlim)

        if ylim is not None:
            plt.ylim(ylim)

        ax.set_zlim(20, 140)
        ax.set_zticks([])

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_title(gene_name)

    def plot_combined_att_img(self, i, channel, time=None, fig=None, axs=None, row=0, 
        titles=False):

        imgs_resized = self.imgs_resized
        
        cur_img = self.trainer.dataloader.dataset.all_imgs_untransformed_channeled[i, channel]
        

        cur_att = self.trainer.collected_attentions[i, channel]


        cur_img_resized = self.trainer.dataloader.dataset.all_imgs[i, channel] #imgs_resized[i, channel]



        combined_img = self.combined_att_imgs[i, channel]
        cols = 5

        if fig is None:
            fig, axs = plt.subplots(1, cols, figsize=(12, 2,))
            fig.tight_layout(rect=[0.0, 0., 1.0, 0.8])
            plt.subplots_adjust(hspace=0.5, wspace=0.25)

        axs[0+(row*cols)].imshow(cur_att, origin='lower', aspect='auto', extent=[-8, 8, 0, 3])
        axs[0+(row*cols)].set_yticks([])
        axs[0+(row*cols)].set_xticks([])
        axs[0+(row*cols)].axvline(0, c='white', lw=1)
        axs[0+(row*cols)].set_ylabel(time)
        if titles:
            axs[0+(row*cols)].set_title("Attentions")

        img = axs[1+(row*cols)].imshow(cur_img, origin='lower', cmap='magma_r',
         aspect='auto', extent=[-8, 8, 0, 3], vmin=0, vmax=0.5)

        axs[1+(row*cols)].set_yticks([])
        axs[1+(row*cols)].set_xticks([])
        axs[1+(row*cols)].axvline(0, c='black', lw=1)

        if titles:
            axs[1+(row*cols)].set_title("Raw data")

        axs[2+(row*cols)].imshow(cur_img_resized, origin='lower', cmap='magma_r', 
            aspect='auto', extent=[-8, 8, 0, 3], vmin=-1, vmax=0)
        axs[2+(row*cols)].set_yticks([])
        axs[2+(row*cols)].set_xticks([])
        axs[2+(row*cols)].axvline(0, c='black', lw=1)
        if titles:
            axs[2+(row*cols)].set_title("Resized raw data")

        from matplotlib.colors import LinearSegmentedColormap
        cm_data = [[0, 0, 0, 1], [0, 0, 0, 0]]
        color_map = LinearSegmentedColormap.from_list('overlay', cm_data)

        axs[3+(row*cols)].imshow(cur_img, origin='lower', cmap='magma_r', aspect='auto', extent=[-8, 8, 0, 3])
        axs[3+(row*cols)].imshow(cur_att, origin='lower', cmap=color_map, aspect='auto', extent=[-8, 8, 0, 3], 
            vmin=0, vmax=1, alpha=0.75)
        axs[3+(row*cols)].set_yticks([])
        axs[3+(row*cols)].set_xticks([])
        axs[3+(row*cols)].axvline(0, c='white', lw=1)
        if titles:
            axs[3+(row*cols)].set_title("Raw data masked\nby attention")

        axs[4+(row*cols)].imshow(combined_img, origin='lower', cmap='viridis', 
            aspect='auto', extent=[-8, 8, 0, 3])
        axs[4+(row*cols)].set_yticks([])
        axs[4+(row*cols)].set_xticks([])
        if titles:
            axs[4+(row*cols)].set_title("Resized raw data\n* attention")
        axs[4+(row*cols)].axvline(0, c='white', lw=1)


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

