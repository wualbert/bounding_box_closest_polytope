import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import Voronoi

def build_centroid_kd_tree(polytopes):
    n = len(polytopes)
    if polytopes[0].type=='AH_polytope':
        k = polytopes[0].t.shape[0]
    elif polytopes[0].type=='zonotope':
        k = polytopes[0].x.shape[0]
    else:
        raise NotImplementedError
    centroids = np.zeros((n,k))
    for i, z in enumerate(polytopes):
        if polytopes[0].type=='AH_polytope':
            centroids[i,:] = polytopes[i].t[:, 0]
        elif polytopes[0].type == 'zonotope':
            centroids[i,:] = polytopes[i].x[:, 0]
        else:
            raise NotImplementedError
    return KDTree(centroids)

def build_AH_polyotpe_centroid_voronoi_diagram(ah_polytopes):
    n = len(ah_polytopes)
    k = ah_polytopes[0].t.shape[0]
    centroids = np.zeros((n, k))
    for i, z in enumerate(ah_polytopes):
        centroids[i, :] = ah_polytopes[i].t[:,0]
    return Voronoi(centroids)
