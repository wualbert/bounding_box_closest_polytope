import numpy as np
from scipy.spatial import cKDTree as KDTree
from pypolycontain.lib.zonotope import zonotope
from collections import deque
from pypolycontain.lib.AH_polytope import AH_polytope,distance_point,to_AH_polytope
from pypolycontain.lib.polytope import polytope
from pypolycontain.lib.zonotope import zonotope_distance_point
from pypolycontain.lib.containment_encodings import subset_generic,constraints_AB_eq_CD,add_Var_matrix
from gurobipy import Model, GRB, QuadExpr

from timeit import default_timer


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

    def build_cell_polytope_map_default(self):
        # compute pairwise distances of the centroids and the polytopes
        for centroid_index, centroid in enumerate(self.centroids):
            centroid_string = str(centroid)
            for polytope_index, polytope in enumerate(self.centroid_to_polytope_map[centroid_string]['polytopes']):
                self.centroid_to_polytope_map[str(centroid)].distances[polytope_index] = distance_point(to_AH_polytope(polytope), centroid)
                # if self.type == 'zonotope':
                #     model = Model("centroid_polytope_distance")
                #     #polytope constraint
                #     n = polytope.x.shape[0]
                #     p = np.empty((polytope.G.shape[1], 1), dtype='object')
                #     polytope_point = np.empty((polytope.x.shape[0], 1), dtype='object')
                #     for row in range(p.shape[0]):
                #         p[row, 0] = model.addVar(lb=-1, ub=1)
                #     for row in range(polytope_point.shape[0]):
                #         polytope_point[row, 0] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
                #     constraints_AB_eq_CD(model, np.eye(n), polytope_point - polytope.x, polytope.G, p)
                #     model.update()
                #     #l2 distance objective
                #     J = QuadExpr()
                #     for dim in range(polytope_point.shape[0]):
                #         J.add((polytope_point[dim,0]-centroid[dim])*(polytope_point[dim,0]-centroid[dim]))
                #     model.setObjective(J, GRB.MINIMIZE)
                #     model.setParam('OutputFlag', 0)
                #     model.update()
                #     model.optimize()
                #     if model.Status == 2:  # found optimal solution
                #         self.centroid_to_polytope_map[str(centroid)].distances[polytope_index] = max(J.getValue(),0.)
                #     else:
                #         print('Warning: Failed to solve minimum distance between polytope and cell. This should never happen.')
                #         print('Failed to solve polytope: ', polytope, 'with centroid at ', polytope.x, 'against Voronoi centroid ', centroid)
                #         print('System will always check for this polytope.')
                #         self.centroid_to_polytope_map[centroid_string].distances[polytope_index] = 0
                # else:
                #     raise NotImplementedError
            self.centroid_to_polytope_map[centroid_string].sort(order='distances')

    def find_closest_polytope(self, query_point, return_intermediate_info = False):
        #find the closest centroid
        d,i = self.centroid_tree.query(query_point)
        closest_centroid = self.centroid_tree.data[i]
        dist_query_centroid = np.linalg.norm(query_point-closest_centroid)
        cutoff_index = np.searchsorted(self.centroid_to_polytope_map[str(closest_centroid)].distances, 2*dist_query_centroid)
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
        if return_intermediate_info:
            return best_polytope, best_distance, closest_polytope_candidates
        return best_polytope