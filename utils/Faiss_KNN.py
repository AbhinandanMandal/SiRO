

import faiss
import numpy as np


"""
===============================================================================
    Class for Fast Similarity Search for classification and retrieval 
    based on k-Nearest Neighbors. This is based on Faiss library 
    Reference: https://github.com/facebookresearch/faiss
===============================================================================
"""

"""
This is custom k-nearestneighbors classifier built using faiss 
for fast similarity search

Basic Idea:
This class -> stores dataset (X,y)
           -> use faiss to quickly find nearest neighbors
           -> predicts labels using majority voting

"""


class FaissKNeighbors:
    def __init__(self, k=5):
        # indexing structure (faiss structure to store vectors)
        self.index = None
        self.y = None  # labels of training data
        self.k = k  # k value for nearest neighbors

    """ Building Faiss index """
    # X,y is the data for training model

    def fit(self, X, y):
        d = X.shape[1]  # no of features per data points (dimension)
        # creating index using L2 (euclidean distance)
        self.index = faiss.IndexFlatL2(d)
        self.index.add(X.astype(np.float32))  # storing all training vectors
        self.y = y  # storing labels

    """
    Finding k-nearest neighbors

    for each input point -> it finds k closest point and return 
    the indices of the nearest neighbors

    for example for any given points -> output is [[2, 5, 1],[0, 3, 4]]
    """

    def neighbors(self, X):
        _, indices = self.index.search(X.astype(np.float32), k=self.k)
        return indices  # indices of the nearest neighbors

    """ Predict label based on majority voting """

    def predict(self, X):
        # indices of the nearest neighbors
        _, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices].astype(np.int32)  # voting
        predictions = np.array([np.argmax(np.bincount(x))
                               for x in votes])  # finding predictions
        return predictions  # returing predictions (array of predicted labels)

    """ Used for SV (single-view) object recognition """

    def predict_SV(self, X):
        _, indices = self.index.search(X.astype(np.float32), k=self.k+1)
        votes = self.y[indices].astype(np.int32)  # voting
        predictions = np.array([np.argmax(np.bincount(x))
                               for x in votes[:, 1:]])
        return predictions

