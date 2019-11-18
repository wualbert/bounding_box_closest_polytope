import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi
from pypolycontain.utils.random_polytope_generator import get_k_random_edge_points_in_zonotope

def build_key_point_kd_tree(polytopes, key_vertex_count = 0, distance_scaling_matrix=None):
    if key_vertex_count > 0:
        n = len(polytopes)*(1+2**key_vertex_count)
    else:
        n = len(polytopes)
    if polytopes[0].__name__=='AH_polytope':
        dim = polytopes[0].t.shape[0]
    elif polytopes[0].__name__=='zonotope':
        dim = polytopes[0].x.shape[0]
    else:
        raise NotImplementedError
    key_point_to_zonotope_map = dict()
    key_points = np.zeros((n,dim))
    for i, p in enumerate(polytopes):
        if p.__name__=='AH_polytope' and key_vertex_count==0:
            key_points[i,:] = p.t[:, 0]
            if distance_scaling_matrix is not None:
                key_points = np.multiply(distance_scaling_matrix, key_points)
            key_point_to_zonotope_map[p.t[:, 0].tostring()]=[p]
        elif p.__name__ == 'zonotope' and key_vertex_count==0:
            key_points[i,:] = p.x[:, 0]
            if distance_scaling_matrix is not None:
                key_points = np.multiply(distance_scaling_matrix, key_points)
            key_point_to_zonotope_map[p.x[:, 0].tostring()]=[p]
        elif p.__name__=='zonotope':
            key_points[i*(1+2**key_vertex_count),:] = p.x[:, 0]
            key_point_to_zonotope_map[p.x[:, 0].tostring()]=[p]
            other_key_points = get_k_random_edge_points_in_zonotope(p, key_vertex_count)
            key_points[i * (2 ** key_vertex_count + 1) + 1:(i + 1) * (2 ** key_vertex_count + 1),
            :] = other_key_points
            if distance_scaling_matrix is not None:
                key_points = np.multiply(distance_scaling_matrix, key_points)
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
