import numpy as np
from scipy.spatial import Voronoi
from pypolycontain.lib.zonotope import zonotope
from collections import deque
from utils.utils import build_centroid_kd_tree, build_polyotpe_centroid_voronoi_diagram
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
        self.centroid_voronoi = build_polyotpe_centroid_voronoi_diagram(self.polytopes)
        print('Built Voronoi diagram in %f seconds' % (default_timer() - self.section_start_time))
        self.section_start_time = default_timer()
        self.centroid_to_polytope_map = dict()  # stores the potential closest polytopes associated with each Voronoi (centroid)
        if preprocess_algorithm != 'default':
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
                    self.centroid_to_polytope_map[hashable] = {p}
        else:
            for centroid in self.centroid_voronoi.points:
                ps = np.asarray(self.polytopes, dtype='object')
                ds = np.zeros(self.polytopes.shape[0])
                self.centroid_to_polytope_map[str(centroid)] = np.rec.fromarrays([ps, ds], names=('polytopes', 'distances'))

        self.centroid_to_voronoi_centroid_index = dict()  # maps each centroid to its internal index in Voronoi

        if preprocess_algorithm == 'vertex':
            self.vertex_to_voronoi_vertex_index = dict()  # maps each vertex to its internal index in Voronoi
            self.vertex_to_voronoi_centroid_index = dict()  # maps each vertex to the list of centroids using it
            self.parse_voronoi_diagram_vertex()
            self.vertex_balls = dict()  # maps each vertex to a ball radius around it
            self.build_sphere_around_vertices()
            print('Computed spheres around vertices in %f seconds' % (default_timer() - self.section_start_time))
            self.section_start_time = default_timer()
            self.build_cell_polytope_map_vertex()
            print('Mapped polytopes in %f seconds' % (default_timer() - self.section_start_time))
            self.section_start_time = default_timer()

        elif preprocess_algorithm=='dmax':
            self.vertex_to_voronoi_vertex_index = dict()  # maps each vertex to its internal index in Voronoi
            self.vertex_to_voronoi_centroid_index = dict()  # maps each vertex to the list of centroids using it
            self.parse_voronoi_diagram_vertex()
            self.cell_dmax = dict()
            self.compute_cell_dmax()
            print('Computed dmax\'s in %f seconds' %(default_timer()-self.section_start_time))
            self.section_start_time = default_timer()
            self.build_cell_polytope_map_dmax()
            print('Mapped polytopes in %f seconds' %(default_timer()-self.section_start_time))
            self.section_start_time = default_timer()

        elif preprocess_algorithm=='default':
            self.build_cell_polytope_map_default()

        else:
            raise NotImplementedError
        #build kd-tree for centroids
        self.centroid_tree = build_centroid_kd_tree(self.polytopes)
        print('Completed precomputation in %f seconds' % (default_timer() - self.init_start_time))

    def build_cell_polytope_map_default(self):
        # compute pairwise distances of the centroids and the polytopes
        for centroid_index, centroid in enumerate(self.centroid_voronoi.points):
            centroid_string = str(centroid)
            for polytope_index, polytope in enumerate(self.centroid_to_polytope_map[centroid_string]['polytopes']):
                if self.type == 'zonotope':
                    model = Model("centroid_polytope_distance")
                    #polytope constraint
                    n = polytope.x.shape[0]
                    p = np.empty((polytope.G.shape[1], 1), dtype='object')
                    polytope_point = np.empty((polytope.x.shape[0], 1), dtype='object')
                    for row in range(p.shape[0]):
                        p[row, 0] = model.addVar(lb=-1, ub=1)
                    for row in range(polytope_point.shape[0]):
                        polytope_point[row, 0] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
                    constraints_AB_eq_CD(model, np.eye(n), polytope_point - polytope.x, polytope.G, p)
                    model.update()
                    #l2 distance objective
                    J = QuadExpr()
                    for dim in range(polytope_point.shape[0]):
                        J.add((polytope_point[dim,0]-centroid[dim])*(polytope_point[dim,0]-centroid[dim]))
                    model.setObjective(J, GRB.MINIMIZE)
                    model.setParam('OutputFlag', 0)
                    model.update()
                    model.optimize()
                    if model.Status == 2:  # found optimal solution
                        self.centroid_to_polytope_map[str(centroid)].distances[polytope_index] = max(J.getValue(),0.)
                    else:
                        print('Warning: Failed to solve minimum distance between polytope and cell. This should never happen.')
                        print('Failed to solve polytope: ', polytope, 'with centroid at ', polytope.x, 'against Voronoi centroid ', centroid)
                        print('System will always check for this polytope.')
                        self.centroid_to_polytope_map[centroid_string].distances[polytope_index] = 0
                else:
                    raise NotImplementedError
            self.centroid_to_polytope_map[centroid_string].sort(order='distances')


    def compute_cell_dmax(self):
        #find the largest distance from the centroid in a voronoi cell
        for centroid_index, centroid in enumerate(self.centroid_voronoi.points):
            vertex_indices = np.asarray(self.get_voronoi_vertex_indices_of_centroid(centroid))
            vertices = self.centroid_voronoi.vertices[vertex_indices]
            norms = np.linalg.norm(np.subtract(vertices,centroid),axis=1)
            current_max = np.amax(norms)
            self.cell_dmax[str(centroid)]=current_max

    def build_cell_polytope_map_dmax(self):
        #for each polytope-voronoi cell combination, check if the cell should be discarded
        for centroid_index, centroid in enumerate(self.centroid_voronoi.points):
            vertex_indices = np.asarray(self.get_voronoi_vertex_indices_of_centroid(centroid))
            vertices = self.centroid_voronoi.vertices[vertex_indices]
            for polytope in self.polytopes:
                if self.type == 'zonotope':
                    model = Model("voronoi_polytope_dmin")
                    # vertex constraint
                    lambda_i = np.empty((1, vertices.shape[0]), dtype='object')
                    for column in range(lambda_i.shape[1]):
                        lambda_i[0, column] = model.addVar(lb=0)
                    constraints_AB_eq_CD(model, lambda_i, np.ones([lambda_i.shape[1], 1]), np.ones([1, 1]),
                                         np.ones([1, 1]))
                    model.update()
                    # convex hull of the Voronoi cell
                    d = vertices.shape[1]
                    voronoi_point = np.empty((d, 1), dtype='object')
                    for row in range(voronoi_point.shape[0]):
                        voronoi_point[row, 0] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
                    constraints_AB_eq_CD(model, np.eye(voronoi_point.shape[0]), voronoi_point, vertices.T, lambda_i.T)
                    model.update()
                    #polytope constraint
                    n = polytope.x.shape[0]
                    p = np.empty((polytope.G.shape[1], 1), dtype='object')
                    polytope_point = np.empty((polytope.x.shape[0], 1), dtype='object')
                    for row in range(p.shape[0]):
                        p[row, 0] = model.addVar(lb=-1, ub=1)
                    for row in range(polytope_point.shape[0]):
                        polytope_point[row, 0] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
                    constraints_AB_eq_CD(model, np.eye(n), polytope_point - polytope.x, polytope.G, p)
                    model.update()
                    #l2 distance objective
                    J = QuadExpr()
                    for dim in range(d):
                        J.add((voronoi_point[dim, 0]-polytope_point[dim,0])* (voronoi_point[dim, 0]-polytope_point[dim,0]))
                    model.setObjective(J, GRB.MINIMIZE)
                    model.setParam('OutputFlag', 0)
                    model.update()
                    model.optimize()
                    if model.Status == 2:  # found optimal solution
                        # print('centroid, polytope: ', centroid, polytope.x)
                        # print('cost ',np.sqrt(max(0.,J.getValue())))
                        if self.cell_dmax[str(centroid)] > np.sqrt(max(0.,J.getValue())):
                            #add the polytope
                            self.centroid_to_polytope_map[str(centroid)].add(polytope)
                    else:
                        print('Warning: Failed to solve minimum distance between polytope and cell. This should never happen.')
                        print('Failed to solve polytope: ', polytope, 'with centroid at ', polytope.x, 'against Voronoi centroid ', centroid)
                        print('System will always check for this polytope.')
                        self.centroid_to_polytope_map[str(centroid)].add(polytope)
                else:
                    raise NotImplementedError

    def parse_voronoi_diagram(self):
        for centroid_index, centroid in enumerate(self.centroid_voronoi.points):
            self.centroid_to_voronoi_centroid_index[str(centroid)] = centroid_index

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
        '''
        Get the vertex indices assiciated with the given centroid
        :param centroid:
        :return:
        '''
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

    def build_cell_polytope_map_vertex(self):
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
                    model.update()
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
        return

    def find_k_closest_polytopes(self,query_point, k_closest=1):
        #find the closest centroid
        d,i = self.centroid_tree.query(query_point)
        closest_voronoi_centroid = self.centroid_tree.data[i]
        closest_AHpolytope_candidates = self.centroid_to_polytope_map[str(closest_voronoi_centroid)]
        #check the AHpolytopes for the closest
        sorted_AHpolytopes = sorted(closest_AHpolytope_candidates, key = lambda p: distance_point(p, query_point))
        return sorted_AHpolytopes[0: min(k_closest, len(sorted_AHpolytopes))]

    def find_closest_polytope(self, query_point, return_intermediate_info = False):
        if self.preprocess_algorithm == 'default':
            #find the closest centroid
            d,i = self.centroid_tree.query(query_point)
            closest_voronoi_centroid = self.centroid_tree.data[i]
            dist_query_centroid = np.linalg.norm(query_point-closest_voronoi_centroid)
            cutoff_index = np.searchsorted(self.centroid_to_polytope_map[str(closest_voronoi_centroid)].distances, 2*dist_query_centroid)
            # print('dqc',dist_query_centroid)
            # print(self.centroid_to_polytope_map[str(closest_voronoi_centroid)].distances)
            closest_polytope_candidates = self.centroid_to_polytope_map[str(closest_voronoi_centroid)].polytopes[0:cutoff_index]
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
        else:
            #find the closest centroid
            d,i = self.centroid_tree.query(query_point)
            closest_voronoi_centroid = self.centroid_tree.data[i]
            closest_polytope_candidates = self.centroid_to_polytope_map[str(closest_voronoi_centroid)]
            #check the AHpolytopes for the closest
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