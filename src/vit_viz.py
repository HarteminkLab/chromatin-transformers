
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import copy
import numpy as np
import cv2

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from matplotlib import collections as mc
from matplotlib.colors import LinearSegmentedColormap
from src.plot_utils import apply_global_settings, plot_density_scatter, plot_rect, hide_spines
from sklearn.metrics import r2_score
import matplotlib.patheffects as path_effects


def plot_gene_tpm(gene_name, vit_data):

    idx = vit_data.index_for(gene_name, time=0.0)
    orf_name = vit_data.orfs[idx]

    tpm_data = vit_data.read_log_tpm_data()

    tpm_pts = tpm_data.loc[orf_name]
    timepoints = tpm_pts.index.values
    xpoints = np.arange(len(timepoints))

    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    ax.plot(xpoints, tpm_pts.values, marker='D', lw=2, c='purple')
    ax.set_xticks(xpoints)
    ax.set_xticklabels([f'{s:.0f}' if s != 7.5 else '7.5' for s in timepoints])
    ax.set_ylim(0, 16)


def plot_gene_prediction(gene_name, time, vit, vit_data, orf_plotter=None, rna_plotter=None,
    discard_ratio=0.95, mask_ratio=0.05, fusion='mean', smooth=False,
    device=torch.device('cpu')):

    apply_global_settings(dpi=80, titlepad=0, linewidth=2)

    idx = vit_data.index_for(gene_name, time=time)
    t = vit_data.times[idx]
    x = vit_data[idx][0].unsqueeze(0)

    orf_name = vit_data.orfs[idx]

    window = 1024
    win_2 = window//2

    gene = vit_data.orfs_data.loc[orf_name]
    span = gene.TSS-win_2, gene.TSS+win_2

    if orf_plotter is not None: orf_plotter.set_span_chrom(span, gene.chr)
    if rna_plotter is not None: rna_plotter.set_span_chrom(span, gene.chr)

    with torch.no_grad():        
        x = x.to(device).float()
        out, weights = vit(x)

    tx = np.log2(vit_data.unscaled_TPM[idx]+1)
    pred_tx = vit_data.unscale_log_tx(out.to(torch.device('cpu'))).item()

    extent = [span[0], span[1], 0, 256]

    title=f"{gene_name}, {str(t)}' True: {tx:.1f}, Pred: {pred_tx:.1f}"

    cm_data = [[0, 0, 0, 1], [0, 0, 0, 0]]
    color_map = LinearSegmentedColormap.from_list('overlay', cm_data)

    resize_size = 256, 1024

    def get_img_and_mask(x, idx):
        att_mask = rollout(vit, x, discard_ratio, 'mean', device=device, attention_channel_idx=idx)
        att_mask[att_mask < mask_ratio] = 0

        img_x = x[idx].cpu().detach().numpy()

        if len(img_x.shape) == 3:
            img_x = img_x[0]
        img_x = cv2.resize(img_x, (resize_size[1], resize_size[0]))

        #att_mask = cv2.resize(att_mask, (resize_size[1], resize_size[0]))
        return img_x, att_mask

    fig, axs = plt.subplots(8, 1, figsize=(8, 14))
    (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7) = axs

    if orf_plotter is not None: orf_plotter.plot_orf_annotations(ax0, flip_genome=(gene.strand == '-'))
    ax0.set_xticks([])

    flip = (gene.strand == '-')

    # TODO: parameterize
    plot_bar_tx = True
    plot_tx_type = 'bar'

    if plot_tx_type == 'transcripts' and rna_plotter is not None: 
    
        rna_plotter.plot(ax1, time, flip_genome=flip)

        negate_scalar = -1 if flip else 1

        ax1.plot(gene.TSS, pred_tx*negate_scalar, marker='x', mfc='red', mew=2, 
            mec=plt.get_cmap('Reds')(0.5), markersize=15)
        ax1.plot(gene.TSS, tx*negate_scalar, marker='o', mfc='none', mew=2, 
            mec=plt.get_cmap('Greens')(0.5),
            markersize=15)

    elif plot_tx_type == 'timecourse':

        tpm_df = vit_data.read_log_tpm_data()
        tpm_course = tpm_df.loc[gene.name]
        print(tpm_course.index.values, tpm_course.values)
        ax1.plot(tpm_course.index.values, tpm_course.values)
        # ax1.xlim
        pass

    elif plot_tx_type == 'bar':
        ax1.axhline(tx, c=plt.get_cmap('Reds')(0.2), linestyle='solid', lw=1, zorder=1)

        path_eff = [path_effects.Stroke(linewidth=2.5, foreground='white'),
                                       path_effects.Normal()]

        ax1.bar(0, tx, width=0.25, color=plt.get_cmap('Reds')(0.5))
        text = ax1.text(0, tx+0.5, f"True: {tx:.1f}", ha='center', 
            fontsize=12, zorder=10)
        text.set_path_effects(path_eff)

        ax1.bar(1, pred_tx, width=0.25, color=plt.get_cmap('Greens')(0.5))
        text = ax1.text(1, pred_tx+0.5, f"Predicted: {pred_tx:.1f}", ha='center', 
            fontsize=12, zorder=10)
        text.set_path_effects(path_eff)

        ax1.set_xlim(-0.5, 1.5)
        ax1.set_ylim(0, 18)
        ax1.set_xticks([])
        ax1.set_yticks([])

    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.925])
    plt.subplots_adjust(hspace=0.2, wspace=0.3)

    def plot_hms(axs, img_x, att_mask):
        ax0, ax1, ax2 = axs

        ax0.imshow(img_x, extent=extent, origin='lower', cmap='magma_r', vmin=-1, vmax=-0.25, 
            aspect='auto')
        ax1.imshow(att_mask, extent=extent, origin='lower', cmap='viridis', vmin=0, vmax=1, 
            aspect='auto')
        ax2.imshow(img_x, extent=extent, origin='lower', cmap='magma_r', vmin=-1, vmax=-0.25, 
            aspect='auto')
        ax2.imshow(att_mask, extent=extent, origin='lower', cmap=color_map, vmin=0, vmax=1, 
            aspect='auto', alpha=0.75)

    if vit.legacy:
        img_x_chan1, att_mask_chan1 = get_img_and_mask(x, 0)
    else:
        img_x_chan1, att_mask_chan1 = get_img_and_mask(x[0], 0)

    plot_hms((ax2, ax3, ax4), img_x_chan1, att_mask_chan1)

    if vit.in_channels == 2:
        img_x_chan2, att_mask_chan2 = get_img_and_mask(x[0], 1)
        plot_hms((ax5, ax6, ax7), img_x_chan2, att_mask_chan2)
    else:
        for ax in [ax5, ax6, ax7]:
            hide_spines(ax)

    xticks = np.arange(gene.TSS-600, gene.TSS+600, 200)
    xticklabels = [str(t) for t in np.arange(-600, 600, 200)]
    xticklabels[3] = 'TSS'

    for ax in axs:
        ax.set_xticks([])
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.patch.set_alpha(0.0)

    ax1.set_ylabel("Transcripts\nlog-TPM", fontsize=18)
    ax3.set_ylabel("Fragment length", fontsize=18)


    ytick_step = 250//3
    for ax in axs[2:5]:
        ax.set_yticks(np.arange(ytick_step//2, 250, ytick_step))
        ax.set_yticklabels(["Small", "Interm.", "Nucl."])

    if vit.in_channels == 1:
        axs = [ax0, ax1, ax2, ax3, ax4]

    for i in range(len(axs)):
        ax = axs[i]
        color = 'gray' if i in [0, 1, 2, 5] else '#dddddd'
        ax.axvline(gene.TSS, lw=1, linestyle='dotted', c=color, alpha=1, zorder=10000)
        ax.set_xticks([])

    if vit.in_channels == 2:
        ax7.set_xticks(xticks)
        ax7.set_xticklabels(xticklabels)
        ax7.set_xlim(*span)
    else:
        ax4.set_xticks(xticks)
        ax4.set_xticklabels(xticklabels)
        ax4.set_xlim(*span)
        ax4.set_xlabel("Genome position (bp), relative to TSS", fontsize=18)

    plt.suptitle(title, fontsize=18)

    return fig


def plot_true_pred_color_tx_0(ret, vit_data, label):
    import pandas as pd
    from src.plot_utils import apply_global_settings
    apply_global_settings()

    unscaled_tx = vit_data.transcription_unscaled
    mean, std = unscaled_tx.mean(), unscaled_tx.std()

    zero_times_ind = np.arange(len(vit_data))[vit_data.orf_times == 0]

    zero_tx = vit_data.transcription_unscaled[zero_times_ind]
    zero_orfs = vit_data.orf_names[zero_times_ind]

    tx_df = pd.DataFrame(index=zero_orfs)
    tx_df['tx_0'] = zero_tx
    tx_df['tx_pred_120'] = ret[1]*std+mean
    tx_df['tx_true_120'] = ret[0]*std+mean

    plt.figure(figsize=(4.75, 4))
    plt.plot([0, 20], [0, 20], lw=1, c='black')
    plt.scatter(tx_df.tx_pred_120, tx_df.tx_true_120, c='none', edgecolor='#c0c0c0', s=3)
    plt.scatter(tx_df.tx_pred_120, tx_df.tx_true_120, c=tx_df.tx_0, cmap='viridis', edgecolor='none', s=3)
    plt.xlim(0, 14)
    plt.ylim(0, 14)
    plt.ylabel("True tx level")
    plt.xlabel(f"{label} predicted tx level")
    plt.title(label, fontsize=24)
    cbar = plt.colorbar()
    cbar.ax.set_title("0 min\nTx level")


def generate_predicted_vs_true_data(vit, vit_data, dataloader, max_num=float('inf'),
        device=torch.device('cpu')):

    all_tx = np.array([])
    all_predictions = np.array([])
    for imgs, tx, _, _, _ in dataloader:    
        with torch.no_grad():
            
            out, weights = vit(imgs.float().to(device))
            predictions = out.detach().to(torch.device('cpu')).numpy().flatten()    
            tx = tx.to(torch.device('cpu'))

            all_tx = np.concatenate([all_tx, tx])
            all_predictions = np.concatenate([all_predictions, predictions])

        # Subsample to save time
        if max_num is not None and len(all_predictions) > max_num:
            break

    r2 = r2_score(all_tx, all_predictions)
    y, x = vit_data.unscale_log_tx(all_tx), vit_data.unscale_log_tx(all_predictions)

    return all_tx, all_predictions, x, y, r2


def plot_predicted_vs_true(vit, vit_data, testloader, max_num=float('inf'), title='', 
        device=torch.device('cpu')):

    all_tx, all_predictions, x, y, r2 = generate_predicted_vs_true_data(vit, 
        vit_data, testloader, max_num=max_num, device=device)

def plot_scatter_predicted_true(x, y):

    ax = plt.gca()

    plot_density_scatter(x, y, cmap='Spectral_r', bw=(0.25, 0.25), zorder=2, ax=ax)

    plt.plot([-20, 20], [-20, 20], c='black', linestyle='solid', lw=0.5, zorder=4)

    plt.xticks(np.arange(0, 20, 5))
    plt.yticks(np.arange(0, 20, 5))
    plt.xlim(-0.5, 15.5)
    plt.ylim(-0.5, 15.5)

    plt.ylabel('True log$_2$ transcript level, TPM')
    plt.xlabel('Predicted log$_2$ transcript level, TPM')

    r2 = r2_score(x, y)
    plt.title(f"n={len(x)}, $R^2$={r2:.3f}")

    # return r2


def plot_loss_progress(loss_df, m):

    def _get_ylim(data):
        data_min = data.min()
        data_max = data.max()
        data_span = data_max-data_min
        return data_min-data_span*0.5, data_max+data_span*0.5

    fig = plt.figure(figsize=(14, 3))
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    plt.subplot(1, 3, 1)
    plt.plot(loss_df.epoch, loss_df.train_loss, label='Training loss')
    plt.plot(loss_df.epoch, loss_df.validation_loss, label='Validation loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(loss_df.epoch[-m:], loss_df.train_loss[-m:], label='Training loss')
    plt.title(f"Training loss, {loss_df.train_loss.values[-1]:.8f}")
    
    plt.subplot(1, 3, 3)
    plt.plot(loss_df.epoch[-m:], loss_df.validation_loss[-m:], label='Validation loss',
        c=plt.get_cmap('tab10')(1))
    
    plt.title(f"Validation loss, {loss_df.validation_loss.values[-1]:.8f}")
    return fig
    

def plot_predictions(vit, vit_data, trainloader, validationloader, testloader, max_num=float('inf'),
        device=torch.device('cpu')):
    fig = plt.figure(figsize=(12, 3.4))
    
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.subplot(1, 3, 1)
    train_r2 = plot_predicted_vs_true(vit, vit_data, trainloader, max_num, 
        title="Training", device=device)

    plt.subplot(1, 3, 2)
    valid_r2 = plot_predicted_vs_true(vit, vit_data, validationloader, max_num,  
        title="Validation", device=device)

    plt.subplot(1, 3, 3)
    test_r2 = plot_predicted_vs_true(vit, vit_data, testloader, max_num,  
        title="Testing", device=device)

    return fig, train_r2, valid_r2, test_r2


def plot_loader_distribution(trainloader, batches=10):
    # Checking training set balance, sampler should evenly collect samples
    # of all transcript levels
    plt.figure(figsize=(3, 2))
    collect = np.array([])
    for j in range(batches):
        for i, data in enumerate(trainloader):
            inputs, tx, _, _, _ = data
            collect = np.append(collect, tx)
    plt.hist(collect)


def create_att_typhoon_plots(gene_name, typhoon_dir, orf_plotter, rna_plotter):
    times = [0, 7.5, 15, 30, 60, 120]

    for time in times:

        fig = plot_gene_prediction(gene_name, time, vit, vit_data, orf_plotter, rna_plotter, smooth=True)
        plt.savefig(f"{typhoon_dir}/{gene_name}_{time}_smooth.png", dpi=150)
        plt.close(fig)
        plt.cla()
        plt.clf()

    for time in times:

        fig = plot_gene_prediction(gene_name, time, vit, vit_data, orf_plotter, rna_plotter, smooth=False)
        plt.savefig(f"{typhoon_dir}/{gene_name}_{time}_raw.png", dpi=150)
        plt.close(fig)
        plt.cla()
        plt.clf()


def visualize_attention(img, vit):

    with torch.no_grad():
        out, att_mat = vit(img.unsqueeze(0))

    att_mat = torch.stack(att_mat).squeeze(1) # combine the 12 layers of attention matrices
    ret_att_mat = copy.deepcopy(att_mat)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    # mask = cv2.resize(mask / mask.max(), (32, 32))[..., np.newaxis]
    mask.shape, img.shape
    #result = (mask * img.moveaxis(0, 2).detach().numpy())
    result = None

    return v, mask, result, ret_att_mat


def rollout(vit, img, discard_ratio=0.95, head_fusion='mean', 
              device=torch.device('cpu'), attention_channel_idx=None):
    
    # Add batch dimensions for vit
    if img.dim() == 3:
        img = img.unsqueeze(0)

    # Assert channels, height, width dimensions
    assert img.dim() == 4
    assert img.shape[0] == vit.in_channels

    out, attentions = vit(img.to(device).float())

    assert attention_channel_idx is not None

    if vit.legacy:
        patch = vit.patches
    else:
        patch = vit.patches[0]
        attentions = attentions[attention_channel_idx]

    rows, cols = patch.patch_rows, patch.patch_cols
    result = torch.eye(attentions[0].size(-1)).to(device)

    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1)).to(device)
            a = (attention_heads_fused + 1.0*I)/2

            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]

    # e.g. In case of 25x100 image with patch size 5x5, 5x20
    mask = mask.reshape(rows, cols).to(torch.device('cpu')).numpy()
    mask = mask / np.max(mask)

    return mask


def resize_masked_image(img, resize_mask):
    resize_img = cv2.resize(img.detach().numpy(), (1024, 256))
    resize_img = (resize_img - resize_img.min()) / (resize_img.max() - resize_img.min()) * 255.
    masked_img = resize_img * resize_mask
    return resize_img, masked_img


def plot_img_masks(x, att_mask, resize_mask):

    fig = plt.figure(figsize=(12, 3))

    resize_img, masked_img = resize_masked_image(x[0, 0], resize_mask)

    extent = [-512, 512, 0, 256]

    plt.subplot(2, 3, 1)
    plt.imshow(resize_img, extent=extent, origin='lower', cmap='Spectral_r', vmax=255, aspect='auto')
    plt.xticks([])

    plt.subplot(2, 3, 3)
    plt.imshow(masked_img, extent=extent, origin='lower', cmap='Spectral_r', vmax=255, aspect='auto')
    plt.xticks([])

    plt.subplot(2, 3, 2)
    plt.imshow(att_mask, extent=extent, origin='lower', vmin=0, vmax=1, cmap='magma', aspect='auto')

    plt.subplot(2, 3, 5)
    plt.imshow(resize_mask, extent=extent, origin='lower', vmin=0, vmax=1, cmap='magma', aspect='auto')

    return fig

