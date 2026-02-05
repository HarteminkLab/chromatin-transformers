# Vision Transformer for Chromatin-Based Gene Expression Prediction

## Overview
This repository implements a vision transformer model that predicts gene transcript levels from chromatin state data. By treating 2D MNase-seq data as images, the model learns regulatory features that drive transcription in an unbiased, data-driven manner.

## Model

<div align="center">
	<img src="https://github.com/HarteminkLab/chromatin-transformers/blob/main/figures/1_model_architecture.jpg" width="70%"/>
  <p style="font-size: 0.9em; line-height: 1.2;">The vision transformer network consists of partitioning the source MNase-seq images into patches, fed into transformer encoder blocks and merged into a feed-forward multilayer perceptron to predict an output transcript level.</p>
</div>

The architecture uses a vision transformer approach that:
Segments MNase-seq chromatin images into patches
Processes patches through transformer encoder blocks to learn attention weights
Identifies critical chromatin features via a feed-forward multilayer perceptron
Predicts transcript-level expression for each gene

The model uses a dual-channel architecture where separate subnetworks process experimental and baseline chromatin states, allowing each channel to learn sample-specific features independently.

## Datasets
Training data comes from a cadmium perturbation time course. For each gene:

Input: 256×1024 bp MNase-seq windows around transcription start sites, downsampled to 32×128 images
Output: Transcripts per million (TPM) values
Baseline channel: Time point 0 minutes provides chromatin state context
Training/validation: Time points 7.5, 15, 30, and 60 minutes
Test set: 120-minute time point (held out)

## Training
The dual-subnetwork architecture processes each input channel independently before merging for final predictions, enabling optimal learning of channel-specific chromatin patterns.


## Results

<div align="center">
	<img src="https://github.com/HarteminkLab/chromatin-transformers/blob/main/figures/2_model_performance.jpg" width="70%"/>
	<p style="font-size: 0.9em; line-height: 1.2;"><bold>(A)</bold> $R^2$ performance of various model architectures and first channels. Transformer network 1 describes networks in which transformer blocks are shared between both input channels. Transformer network 2 describes networks in which each channels uses its own subnetwork to learn appropriate channel-specific attention weights. Each network type was trained against different initial channels: 0 minutes, 7.5 minutes, or a wild-type from an entirely different time course. The best performing models use two different subnetworks for each channel and input data from the cadmium time course as the first channel. <bold>(B)</bold> Results from the best performing network show a coefficient of determination value, $R^2$ of 0.709. </p>
</div>

The best performing model achieved R² = 0.709 on the held-out test set, substantially outperforming the baseline (R² = 0.44). Learned attention weights successfully identify predictive chromatin features and can be visualized to understand which genomic regions most influence transcriptional output.

## Attention Weight Analysis

<div align="center">
	<img src="https://github.com/HarteminkLab/chromatin-transformers/blob/main/figures/3_learned_attentions.jpg" width="40%"/>
	<p style="font-size: 0.9em; line-height: 1.2;">
		Learned attention weights identify features of the chromatin most informative in predicting transcription. Row 2 shows the predicted transcript level (second row; red X) matches well with the true transcript level (second row; green circle). Rows 3-5 depict the source MNase-seq and attention weights for 0 minute sample for <em>PDC6</em>, Rows 6-8 depict the source MNase-seq and attention weights for 120 minute sample for <em>PDC6</em>.
	</p>
</div>

The transformer architecture enables direct interpretation of predictive chromatin features through attention weight visualization. For each gene, the model's attention weights can be extracted and merged to highlight the most informative patches in the input MNase-seq images. 

This reveals which specific chromatin regions drive transcriptional predictions. These highlighted regions can then be subjected to motif and sequence analysis to discover transcription factor binding sites and other small regulatory elements in gene promoters, providing biological insight into the chromatin features that govern gene expression.
