import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import Voronoi

def build_centroid_kd_tree(zonotopes):
    n = len(zonotopes)
    k = zonotopes[0].x.shape[0]
    centroids = np.zeros((n,k))
    for i, z in enumerate(zonotopes):
        centroids[i,:] = zonotopes[i].x[:,0]
    return KDTree(centroids)

def build_centroid_voronoi_diagram(zonotopes):
    n = len(zonotopes)
    k = zonotopes[0].x.shape[0]
    centroids = np.zeros((n, k))
    for i, z in enumerate(zonotopes):
        centroids[i, :] = zonotopes[i].x[:, 0]
    return Voronoi(centroids)
