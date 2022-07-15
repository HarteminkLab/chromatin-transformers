

import sys
sys.path.append('.')

from src.reference_data import read_chromosome_span
from src.utils import print_fl
from src.math_utils import nearest_span
from src.utils import mkdirs_safe
import matplotlib.pyplot as plt

from src.plot_utils import plot_density_scatter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.transformations import exhaustive_counts
from src.chromatin import filter_mnase
from src.orf_plotter import ORFAnnotationPlotter
from src.plot_utils import hide_spines


class TyphoonPlotter:

    def __init__(self, all_mnase_data=None, orf_plotter=None):

        if all_mnase_data is None:
            self.all_mnase_data = pd.read_hdf('data/cac1_mnase_sampled_merged.h5.z', 
                'data')
            self.orf_plotter = orf_plotter
        else:
            self.all_mnase_data = all_mnase_data
            self.orf_plotter = ORFAnnotationPlotter()

        self.times = [0.0, 7.5, 15, 30, 60, 120]
        self.chrom = None
        self.span = None


    def select_gene(self, gene_name, padding=1500):
        orfs = self.orf_plotter.orfs
        gene = orfs[orfs['name'] == gene_name].reset_index(drop=False).loc[0]
        span = gene.TSS-padding, gene.TSS+padding
        self.select_data(span, gene.chr)


    def select_data(self, span, chrom, time=None):

        if (chrom == self.chrom and span[0] == self.span[0] and 
            span[1] == self.span[1] and time == self.time):
            return

        length_span = 100, 200

        data = self.all_mnase_data
        self.time = time
        self.length_span = length_span
        self.span = span
        self.chrom = chrom
        data = data[(data.chr == chrom) & (data.mid > span[0]) & 
                    (data.mid < span[1])]

        if time is not None:
            data = data[(data.time == time)]

        self.current_mnase = data
        # self.data_hist = exhaustive_counts(data, span, (0, 250))
        self.orf_plotter.set_span_chrom(span, chrom)


    def plot_data(self, figsize=(12, 12), times=None, title=None, titles=True, labels=True, ticks=True):

        if times is None:
            times = self.times

        fig, axs = plt.subplots(7, 1, figsize=figsize)

        # Left, Bottom, Right, Top
        fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        self.orf_plotter.plot_orf_annotations(axs[0])

        span = self.span

        for i in range(len(times)):
            ax = axs[i+1]

            time = times[i]
            plot_data = self.current_mnase
            plot_data = plot_data[plot_data.time == time]
            plot_density_scatter(plot_data['mid'].values, 
                                 plot_data['length'].values, bw=(5, 15), ax=ax,
                                 vmax=0.00004)

            if titles:
                ax.set_title(f"{str(time)} minutes", fontsize=16)
            ax.set_xticks(np.arange(span[0], span[1]+500, 2000), minor=False)
            ax.set_xticks(np.arange(span[0], span[1]+500, 1000), minor=True)
            ax.set_xlim(*self.span)

            ax.set_xlim(span[0], span[1])

            if ticks:
                ax.set_yticks([0, 100, 200])
                ax.set_ylim(0, 250)

        if not ticks:
            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])

        for ax in axs[len(times)+1:]:
            hide_spines(ax)

        if labels:
            axs[len(times)//2+1].set_ylabel("Fragment length, bp", fontsize=12)
            axs[len(times)].set_xlabel("Genomic position, bp", fontsize=12)

        if title is None:
            title = f"chr{self.chrom}, {self.span[0]:.0f}...{self.span[1]:.0f}"

        plt.suptitle(title, fontsize=24)

        return axs
