
# main idea of pose-invariant losses are dependent on euclidean distances
# building euclidean distances for embeddings x and y

import torch
import matplotlib.pyplot as plt
import numpy as np 
from csv import writer
import random

def expand_pairwise_distances(x,y=None): # euclidean distances between the embeddings x, y
    # dist[i,j] = ||xi - yj||^2
    # x is Nxd matrix, y is Mxd matrix (optional)
    # dist shape NxM 
    # explaination
    # x:(N,d), y:(M,d)
    # x.unsqueeze(1) = x(N,1,d)
    # y.unsqueeze(0) = y(1,M,d)
    # x[i] - y[j] = (N,M,d)
    # returns embeddings of shape (N,M)
    
    if y is not None:
        differences = x.unsqueeze(1) - y.unsqueeze(0) # when x,y both embeddings
    else:
        differences = x.unsqueeze(1) - x.unsqueeze(0) # only x embedding is present
    distances = torch.sum(differences*differences -1) # summing the differences
    return distances # euclidean distance between x, y



