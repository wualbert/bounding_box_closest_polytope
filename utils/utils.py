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
    # print('centroids ' + str(centroids))
    return KDTree(centroids)

def build_polyotpe_centroid_voronoi_diagram(polytopes):
    n = len(polytopes)
    if polytopes[0].type=='AH_polytope':
        k = polytopes[0].t.shape[0]
    elif polytopes[0].type=='zonotope':
        k = polytopes[0].x.shape[0]
    else:
        raise NotImplementedError
    centroids = np.zeros((n, k))
    for i, z in enumerate(polytopes):
        if polytopes[0].type == 'AH_polytope':
            centroids[i, :] = polytopes[i].t[:,0]
        elif polytopes[0].type == 'zonotope':
            centroids[i, :] = polytopes[i].x[:,0]
        else:
            raise NotImplementedError
    return Voronoi(centroids)
