import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi
from pypolycontain.utils.random_polytope_generator import get_k_random_edge_points_in_zonotope

def build_key_point_kd_tree(polytopes, key_vertex_count = 2):
    n = len(polytopes)*(1+2**key_vertex_count)
    if polytopes[0].type=='AH_polytope':
        k = polytopes[0].t.shape[0]
    elif polytopes[0].type=='zonotope':
        k = polytopes[0].x.shape[0]
    else:
        raise NotImplementedError
    key_point_to_zonotope_map = dict()
    key_points = np.zeros((n,k))
    for i, p in enumerate(polytopes):
        if p.type=='AH_polytope' and k==0:
            key_points[i,:] = p.t[:, 0]
            key_point_to_zonotope_map[p.t[:, 0].tostring()]=[p]
        elif p.type == 'zonotope' and k==0:
            key_points[i,:] = p.x[:, 0]
            key_point_to_zonotope_map[p.x[:, 0].tostring()]=[p]
        elif p.type=='zonotope':
            key_points[i*(1+2**key_vertex_count),:] = p.x[:, 0]
            key_point_to_zonotope_map[p.x[:, 0].tostring()]=[p]
            other_key_points = get_k_random_edge_points_in_zonotope(p, key_vertex_count)
            # print(other_key_points.shape)
            # print(other_key_points)
            key_points[i * (2 ** key_vertex_count + 1) + 1:(i + 1) * (2 ** key_vertex_count + 1),
            :] = other_key_points
            for kp in other_key_points:
                key_point_to_zonotope_map[kp.tostring()] = [p]
        else:
            raise NotImplementedError
    return KDTree(key_points),key_point_to_zonotope_map

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
