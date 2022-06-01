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
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

# Base ViT pytorch implementation
# https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632

# Attention capture and visualization
# https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb

class PatchEmbedding(nn.Module):
    """
    Converts a batch of images into patch embeddings including CLS token
    and position embeddings
    """
    def __init__(self, img_size: (int, int), in_channels: int, patch_size: int, 
        emb_size: int):

        super().__init__()

        self.patch_size = patch_size
        self.emb_size = emb_size
        self.patch_rows, self.patch_cols = (img_size[0]//patch_size), \
                                           (img_size[1]//patch_size)

        # Partition the source image into patches using a convolution and rearrange
        # the dimensions into (batches, num patches, embeddings)
        self.patches = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, 
            stride=patch_size)
        self.rearrange = Rearrange('b e (r) (c) -> b (r c) e')

        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))

        # Number of patches + CLS token
        self.num_params = self.patch_rows*self.patch_cols+1

        # Position embeddings, add to each parameter
        self.positions = nn.Parameter(torch.randn(self.num_params, emb_size))

    def forward(self, x, **kwargs):
        b, _, _, _ = x.shape

        x = self.patches(x)
        x = self.rearrange(x) # (batches, num patches, embeddings)

        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)

        # Prepend the cls token to the input and add the position embeddings
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, 
                       num_heads: int = 8, 
                       attention_dropout: float = 0, 
                       projection_dropout: float = 0):
        super().__init__()

        self.emb_size = emb_size
        self.num_heads = num_heads

        self.num_attention_heads = num_heads
        self.attention_head_size = int(emb_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(emb_size, self.all_head_size)
        self.key = nn.Linear(emb_size, self.all_head_size)
        self.value = nn.Linear(emb_size, self.all_head_size)

        self.out = nn.Linear(emb_size, emb_size)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(projection_dropout)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        """
        Rearrange (batches, patches, (number of heads * head size) into
        (batches, number of heads, patches, head size)

        * Note that number of embeddings is divisible by the number of heads
        """
        x = rearrange(x, "b n (h d) -> b h n d", h=self.num_attention_heads)
        return x

    def forward(self, x, **kwargs):
        """Compute self-attention"""
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Compute self-attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        # Multiply attention proabilities against values to compute context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_size: int,
                 forward_expansion: int,
                 num_heads: int,
                 att_drop_p: float = 0.,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__()

        self.attention_norm = nn.LayerNorm(emb_size)
        self.attention = MultiHeadAttention(emb_size, 
            num_heads, att_drop_p, forward_drop_p, **kwargs)
        self.feed_forward_norm = nn.LayerNorm(emb_size)
        self.feed_forward = FeedForwardBlock(emb_size,
            expansion=forward_expansion, drop_p=forward_drop_p)

    def forward(self, x, **kwargs):
        
        # Residually add the attention output
        h = x
        x = self.attention_norm(x)
        x, weights = self.attention(x)
        x = x + h

        # Residual add feed forward
        h = x
        x = self.feed_forward_norm(x)
        x = self.feed_forward(x)
        x = x + h

        return x, weights


class TransformerEncoder(nn.Module):
    def __init__(self, emb_size: int,
                       num_heads: int = 8,
                       transformer_depth: int = 12,
                       forward_expansion: int = 4,
                       att_drop_p: float = 0.,
                       forward_drop_p: float = 0.,
                       **kwargs):

        super().__init__()

        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(emb_size)
        for _ in range(transformer_depth):
            layer = TransformerEncoderBlock(emb_size=emb_size,
                 forward_expansion=forward_expansion,
                 num_heads=num_heads,
                 att_drop_p=att_drop_p,
                 forward_drop_p=forward_drop_p)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x, **kwargs):
        attn_weights = []
        for layer_block in self.layer:
            x, weights = layer_block(x)
            attn_weights.append(weights)
        encoded = self.encoder_norm(x)
        return encoded, attn_weights


class MergeHead(nn.Module):
    def __init__(self, in_channels: int, num_params: int, emb_size: int, 
        n_classes: int, depth: int = 4, drop_p: float = 0.):
        super().__init__()

        self.layers = nn.ModuleList()
        # Number of images in batch, input channels, 
        # number of parameters/patches, number of embeddings
        self.rearrange = Rearrange('b i n e -> b (i n e)')

        # Merge per channel layers into 1
        layer = nn.Linear(in_channels*num_params*emb_size, emb_size)
        self.layers.append(copy.deepcopy(layer))

        # Forward through MLP
        for _ in range(depth):
            layer_norm = nn.LayerNorm(emb_size)
            layer = nn.Linear(emb_size, emb_size)
            self.layers.append(copy.deepcopy(layer_norm))
            self.layers.append(copy.deepcopy(layer))

        layer_norm = nn.LayerNorm(emb_size)
        layer = nn.Linear(emb_size, n_classes)
        self.layers.append(copy.deepcopy(layer_norm))
        self.layers.append(copy.deepcopy(layer))

    def forward(self, x, **kwargs):
        x = self.rearrange(x)
        for layer_block in self.layers:
            x = layer_block(x)
        return x


class ViT(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        self.config = config
        self.n_classes = config.NUM_CLASSES
        self.in_channels = config.IN_CHANNELS
        self.img_size = config.IMG_SIZE
        self.patch_size = config.PATCH_SIZE
        self.emb_size = config.EMB_SIZE
        self.num_heads = config.NUM_HEADS
        self.transformer_depth = config.DEPTH
        self.forward_expansion = config.FORWARD_EXPANSION
        self.att_drop_p = config.DROPOUT
        self.forward_drop_p = config.DROPOUT

        # Patch Embeddings and encoders per channel
        self.patches = nn.ModuleList()
        self.encoders = nn.ModuleList()
        for _ in range(self.in_channels):

            patch = PatchEmbedding(in_channels=1,
                patch_size=self.patch_size, 
                img_size=self.img_size, 
                emb_size=self.emb_size)
            encoder = TransformerEncoder(emb_size=self.emb_size, 
                                              num_heads=self.num_heads,
                                              transformer_depth=self.transformer_depth, 
                                              forward_expansion=self.forward_expansion,
                                              att_drop_p=self.att_drop_p,
                                              forward_drop_p=self.forward_drop_p,
                                               **kwargs)
            self.patches.append(copy.deepcopy(patch))
            self.encoders.append(copy.deepcopy(encoder))

        self.num_params = patch.num_params
        self.merge_head = MergeHead(in_channels=self.in_channels,
            n_classes=self.n_classes, num_params=self.num_params, 
            emb_size=self.emb_size, depth=4, drop_p=self.forward_drop_p)

    def config_repr(self):

        return (f"n_classes: {self.n_classes}\n"
                f"in_channels: {self.in_channels}\n"
                f"img_size: {self.img_size}\n"
                f"patch_size: {self.patch_size}\n"
                f"emb_size: {self.emb_size}\n"
                f"num_heads: {self.num_heads}\n"
                f"transformer_depth: {self.transformer_depth}\n"
                f"forward_expansion: {self.forward_expansion}\n"
                f"att_drop_p: {self.att_drop_p}\n"
                f"forward_drop_p: {self.forward_drop_p}\n")

    def forward(self, in_x, **kwargs):

        # Matrix to hold output from each channel's network
        device = next(self.patches[0].parameters()).device

        out = torch.zeros((in_x.shape[0], self.in_channels, 
            self.num_params, self.emb_size)).to(device)
        out_weights = []

        for i in range(self.in_channels):
            x = self.patches[i](in_x[:, i:i+1])
            x, weights = self.encoders[i](x)
            out[:, i] = x
            out_weights.append(weights)

        # Merge the output from each channel's network
        out = self.merge_head(out)
        return out, out_weights
