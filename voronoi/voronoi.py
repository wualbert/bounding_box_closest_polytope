import numpy as np
from scipy.spatial import cKDTree as KDTree
from pypolycontain.lib.zonotope import zonotope
from collections import deque
from pypolycontain.lib.AH_polytope import AH_polytope,distance_point,to_AH_polytope
from pypolycontain.lib.polytope import polytope
from pypolycontain.lib.zonotope import zonotope_distance_point
from pypolycontain.lib.containment_encodings import subset_generic,constraints_AB_eq_CD,add_Var_matrix
from gurobipy import Model, GRB, QuadExpr

import itertools
from multiprocessing import Pool, Array

from timeit import default_timer


def set_polytope_pair_distance(arguments):
    centroids, centroid_to_polytope_map, polytope_index, centroid_index = arguments
    centroid = centroids[centroid_index]
    centroid_string = str(centroid)
    polytope = centroid_to_polytope_map[centroid_string]['polytopes'][polytope_index]
    return distance_point(to_AH_polytope(polytope), centroid)

class VoronoiClosestPolytope:
    def __init__(self, polytopes, preprocess_algorithm = 'default'):
        '''
        Compute the closest polytope using Voronoi cells
        :param polytopes:
        :param preprocess_algorithm: 'default', 'vertex', or 'dmax'
        '''
        self.preprocess_algorithm = preprocess_algorithm
        self.init_start_time = default_timer()
        self.section_start_time = self.init_start_time
        self.polytopes = np.asarray(polytopes, dtype='object')
        self.type = self.polytopes[0].type
        if self.type == 'AH_polytope':
            self.dim = self.polytopes[0].t.shape[0]
        elif self.type == 'zonotope':
            self.dim =self.polytopes[0].x.shape[0]
        else:
            raise NotImplementedError
        self.centroids = np.zeros([len(self.polytopes), self.dim])
        for i, z in enumerate(polytopes):
            if self.type == 'AH_polytope':
                self.centroids[i, :] = self.polytopes[i].t[:, 0]
            elif self.type == 'zonotope':
                self.centroids[i, :] = self.polytopes[i].x[:, 0]
            else:
                raise NotImplementedError

        self.centroid_to_polytope_map = dict()  # stores the potential closest polytopes associated with each Voronoi (centroid)
        for centroid in self.centroids:
            ds = np.zeros(self.polytopes.shape[0])
            self.centroid_to_polytope_map[str(centroid)] = np.rec.fromarrays([self.polytopes, ds], names=('polytopes', 'distances'))

        self.build_cell_polytope_map_default()

        #build kd-tree for centroids
        self.centroid_tree = KDTree(self.centroids)
        print('Completed precomputation in %f seconds' % (default_timer() - self.init_start_time))

    def build_cell_polytope_map_default(self, process_count = 8):
        polytope_centroid_indices = np.array(np.meshgrid(np.arange(self.polytopes.shape[0]),np.arange(self.centroids.shape[0]))).T.reshape(-1, 2)
        arguments = []
        for i in polytope_centroid_indices:
            arguments.append((self.centroids, self.centroid_to_polytope_map, i[0],i[1]))
        p = Pool(process_count)
        pca = p.map(set_polytope_pair_distance, arguments)
        polytope_centroid_arrays=np.asarray(pca).reshape((self.polytopes.shape[0]),self.centroids.shape[0])
        # print(polytope_centroid_arrays)
        # compute pairwise distances of the centroids and the polytopes
        #fixme
        for centroid_index, centroid in enumerate(self.centroids):
            centroid_string = str(centroid)
            for polytope_index, polytope in enumerate(self.centroid_to_polytope_map[centroid_string]['polytopes']):
                self.centroid_to_polytope_map[str(centroid)].distances[polytope_index] = polytope_centroid_arrays[polytope_index, centroid_index]
            self.centroid_to_polytope_map[centroid_string].sort(order='distances')
            # print(self.centroid_to_polytope_map[centroid_string])

    def find_closest_polytope(self, query_point, return_intermediate_info = False):
        #find the closest centroid
        d,i = self.centroid_tree.query(query_point)
        closest_centroid = self.centroid_tree.data[i]
        # print(closest_centroid)
        closest_centroid_polytope = self.centroid_to_polytope_map[str(closest_centroid)]['polytopes'][0]
        dist_query_centroid_polytope = distance_point(closest_centroid_polytope, query_point)
        dist_query_centroid = np.linalg.norm(query_point-closest_centroid)
        # print(dist_query_centroid, dist_query_centroid_polytope)
        cutoff_index = np.searchsorted(self.centroid_to_polytope_map[str(closest_centroid)].distances, dist_query_centroid+dist_query_centroid_polytope)
        # print(cutoff_index)
        # print(self.centroid_to_polytope_map[str(closest_centroid)]['distances'][0:cutoff_index])
        # print(self.centroid_to_polytope_map[str(closest_centroid)]['distances'][cutoff_index:])
        # print('dqc',dist_query_centroid)
        # print(self.centroid_to_polytope_map[str(closest_centroid)].distances)
        closest_polytope_candidates = self.centroid_to_polytope_map[str(closest_centroid)].polytopes[0:cutoff_index]
        # print(closest_polytope_candidates)
        best_polytope = None
        best_distance = np.inf
        for polytope in closest_polytope_candidates:
            dist = distance_point(polytope, query_point)
            if best_distance>dist:
                best_distance = dist
                best_polytope = polytope
        # print('best distance', best_distance)
        if return_intermediate_info:
            return best_polytope, best_distance, closest_polytope_candidates
        return best_polytope