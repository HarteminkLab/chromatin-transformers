# Vision Transformer for Chromatin-Based Gene Expression Prediction

## Overview
This repository implements a vision transformer model that predicts gene transcript levels from chromatin state data. By treating 2D MNase-seq data as images, the model learns regulatory features that drive transcription in an unbiased, data-driven manner.

## Model
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
The best performing model achieved R² = 0.709 on the held-out test set, substantially outperforming the baseline (R² = 0.44). Learned attention weights successfully identify predictive chromatin features and can be visualized to understand which genomic regions most influence transcriptional output.
