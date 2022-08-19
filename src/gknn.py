"""
Modified from:

Generalized k-nearest neighbor entropy estimation
authors: Niru Maheswaranathan and Lane McIntosh
04:16 PM Apr 7, 2014

Mnatsakanov, et, al 2008, k_n Nearest Neighbor Estimators of Entropy

https://github.com/baccuslab/gknn-entropy
"""

import numpy as np
from math import gamma
from sklearn.metrics import DistanceMetric

def entropy(data, ball='euclidean', k=1):
    
    # Get number of samples and dimensionality
    (n, p)  = data.shape
    
    b = 2 # euclidean distance
        
    # Get pairwise distances, and sort
    dist  = DistanceMetric.get_metric('euclidean')
    D_mat = dist.pairwise(data)
    D_mat.sort(axis=1)
    radii = D_mat[:, k]

    # compute volumes of each p-dimensional ball for each entry
    volumes = (2*gamma(1/b + 1)*radii)**p / gamma(p/b + 1)
    
    # Compute estimated entropy for volumes
    return sum([np.log2(vol) for vol in volumes])/float(n) + np.log2(n) - L(k - 1) + 0.577215665


def L(k):
    if k == 0:
        return 0
    elif k > 0:
        return sum([1/float(i) for i in range(1, k+1)])
