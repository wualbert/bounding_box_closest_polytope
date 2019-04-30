import numpy as np
from scipy.spatial import Voronoi
from pypolycontain.lib.zonotope import zonotope
from collections import deque
from utils.utils import build_centroid_kd_tree, build_polyotpe_centroid_voronoi_diagram
from pypolycontain.lib.AH_polytope import AH_polytope,distance_point,to_AH_polytope
from pypolycontain.lib.polytope import polytope
from pypolycontain.lib.zonotope import zonotope_distance_point
from pypolycontain.lib.containment_encodings import subset_generic,constraints_AB_eq_CD,add_Var_matrix
from gurobipy import Model, GRB

from timeit import default_timer


class VoronoiClosestPolytope:
    def __init__(self, polytopes, compute_with_vertex = False):
        self.start_time = default_timer()
        self.polytopes = polytopes
        self.type = self.polytopes[0].type
        self.centroid_voronoi = build_polyotpe_centroid_voronoi_diagram(self.polytopes)
        self.centroid_to_polytope_map = dict()  #stores the potential closest polytopes associated with each Voronoi (centroid)
        for p in self.polytopes:
            if self.type == 'AH_polytope':
                hashable = str(p.t.flatten())
            elif self.type == 'zonotope':
                hashable = str(p.x.flatten())
            else:
                raise NotImplementedError
            if hashable in self.centroid_to_polytope_map:
                print('Warning: found polytope with identical centroids!')
                self.centroid_to_polytope_map[hashable].add(p)
            else:
                self.centroid_to_polytope_map[hashable]={p}
        if compute_with_vertex:
            self.centroid_to_voronoi_centroid_index = dict()  # maps each centroid to its internal index in Voronoi
            self.vertex_to_voronoi_vertex_index = dict()  # maps each vertex to its internal index in Voronoi
            self.vertex_to_voronoi_centroid_index = dict()  # maps each vertex to the list of centroids using it
            self.vertex_balls = dict()  # maps each vertex to a ball radius around it
            self.parse_voronoi_diagram_vertex()
            self.build_sphere_around_vertices()
            self.build_cell_AHpolytope_map_vertex()

        #build kd-tree for centroids
        self.centroid_tree = build_centroid_kd_tree(self.polytopes)

    def parse_voronoi_diagram_vertex(self):
        for centroid_index, centroid in enumerate(self.centroid_voronoi.points):
            self.centroid_to_voronoi_centroid_index[str(centroid)] = centroid_index
        for vertex_index, vertex in enumerate(self.centroid_voronoi.vertices):
            self.vertex_to_voronoi_vertex_index[str(vertex)] = vertex_index
        for centroid in self.centroid_voronoi.points:
            vertex_indices = self.get_voronoi_vertex_indices_of_centroid(centroid)
            # print(centroid, vertex_indices.shape)
            for vertex_index in np.atleast_1d(vertex_indices):
                vertex = self.centroid_voronoi.vertices[vertex_index]
                hashable = str(vertex)
                if hashable not in self.vertex_to_voronoi_centroid_index:
                    self.vertex_to_voronoi_centroid_index[hashable] = deque()
                centroid_index = self.centroid_to_voronoi_centroid_index[str(centroid)]
                self.vertex_to_voronoi_centroid_index[hashable].append(centroid_index)
        return

    def get_voronoi_vertex_indices_of_centroid(self, centroid):
        assert(str(centroid) in self.centroid_to_voronoi_centroid_index)
        region_index = self.centroid_voronoi.point_region[self.centroid_to_voronoi_centroid_index[str(centroid)]]
        vertex_indices = np.asarray(self.centroid_voronoi.regions[region_index])
        valid_vertex_indices = vertex_indices[np.where(vertex_indices!=-1)]

        return valid_vertex_indices

    def build_sphere_around_vertices(self):
        for vertex in self.centroid_voronoi.vertices:
            closest_centroid_index = self.vertex_to_voronoi_centroid_index[str(vertex)][0]
            # print('cci', closest_centroid_index)
            # print('centroid', self.centroid_voronoi.points[closest_centroid_index])
            vertex_to_centroid_distance = np.linalg.norm(vertex-self.centroid_voronoi.points[closest_centroid_index])
            #construct ball around the vertex
            self.vertex_balls[str(vertex)] = vertex_to_centroid_distance
        return

    def build_cell_AHpolytope_map_vertex(self):
        # check Voronoi overlap with polytope through convex hull
        for centroid in self.centroid_voronoi.points:
            vertex_ids = self.get_voronoi_vertex_indices_of_centroid(centroid)
            vertices = self.centroid_voronoi.vertices[vertex_ids]
            # linear program for checking feasibility
            # model.setParam('OutputFlag', False)
            for polytope in self.polytopes:
                #check each vertex and see if it is contained by the polytope
                #zonotope case
                if self.type=='zonotope':
                    model = Model("polytope_in_voronoi")
                    n = polytope.x.shape[0]
                    p = np.empty((polytope.G.shape[1], 1), dtype='object')
                    x = np.empty((polytope.x.shape[0], 1), dtype='object')
                    for row in range(p.shape[0]):
                        p[row, 0] = model.addVar(lb=-1, ub=1)
                    model.update()
                    # vertex constraint
                    lambda_i = np.empty((1, vertices.shape[0]), dtype='object')
                    for column in range(lambda_i.shape[1]):
                        lambda_i[0, column] = model.addVar(lb=0)
                    constraints_AB_eq_CD(model, lambda_i, np.ones([lambda_i.shape[1], 1]), np.ones([1, 1]), np.ones([1, 1]))
                    model.update()
                    #constrain the point to be inside the polytope
                    for row in range(x.shape[0]):
                        x[row, 0] = model.addVar()
                    constraints_AB_eq_CD(model, np.eye(n), x - polytope.x, polytope.G, p)
                    constraints_AB_eq_CD(model, np.eye(x.shape[0]), x, vertices.T, lambda_i.T)
                    model.setParam('OutputFlag', 0)
                    model.optimize()
                    if model.Status == 2: #found optimal solution
                        self.centroid_to_polytope_map[str(centroid)].add(polytope)
                elif self.type=='AH_polytope':
                    raise NotImplementedError
                else:
                    raise NotImplementedError

        for polytope in self.polytopes:
            for vertex in self.centroid_voronoi.vertices:
                if self.type == 'zonotope':
                    dist = zonotope_distance_point(polytope, vertex)
                    if dist <= self.vertex_balls[str(vertex)]:
                        associated_centroid_ids = self.vertex_to_voronoi_centroid_index[str(vertex)]
                        for centroid_id in associated_centroid_ids:
                            centroid = self.centroid_voronoi.points[centroid_id]
                            self.centroid_to_polytope_map[str(centroid)].add(polytope)
                elif self.type=='AH_polytope':
                    dist = distance_point(polytope, vertex)
                    if dist <= self.vertex_balls[str(vertex)]:
                        associated_centroid_ids = self.vertex_to_voronoi_centroid_index[str(vertex)]
                        for centroid_id in associated_centroid_ids:
                            centroid = self.centroid_voronoi.points[centroid_id]
                            self.centroid_to_polytope_map[str(centroid)].add(polytope)
        print('Completed precomputation in %f seconds' %(default_timer()-self.start_time))
        return

    def find_closest_polytopes(self, query_point, k_closest = 1):
        #find the closest centroid
        start_time = default_timer()
        d,i = self.centroid_tree.query(query_point)
        closest_voronoi_centroid = self.centroid_tree.data[i]
        closest_AHpolytope_candidates = self.centroid_to_polytope_map[str(closest_voronoi_centroid)]
        #check the AHpolytopes for the closest
        sorted_AHpolytopes = sorted(closest_AHpolytope_candidates, key = lambda p: distance_point(p, query_point))
        # print('Found closest polytope in %f seconds' %(default_timer()-start_time))
        return sorted_AHpolytopes[0: min(k_closest, len(sorted_AHpolytopes))]