import numpy as np
from scipy.spatial import Voronoi
from pypolycontain.lib.zonotope import zonotope
from collections import deque
from utils.utils import build_centroid_kd_tree, build_AH_polyotpe_centroid_voronoi_diagram
from pypolycontain.lib.AH_polytope import AH_polytope,distance_point,to_AH_polytope
from pypolycontain.lib.polytope import polytope


# class VoronoiCell:
#     def __init__(self, centroid, vertices, ridges):
#         self.centroid = centroid
#         self.vertices = vertices
#         self.polytopes = deque()
#         self.id = hash(str(self.centroid))
#
#     def add_polytopes(self, polytopes):
#         self.polytopes.append(polytopes)
#
#     def __hash__(self):
#         return self.id

class VoronoiClosestAHPolytope:
    def __init__(self, AHpolytopes):
        self.AHpolytopes = AHpolytopes
        self.centroid_voronoi = build_AH_polyotpe_centroid_voronoi_diagram(self.AHpolytopes)
        self.centroid_to_AHpolytope_map = dict()  #stores the potential closest polytopes associated with each Voronoi (centroid)
        for p in self.AHpolytopes:
            hashable = str(p.t.flatten())
            if hashable in self.centroid_to_AHpolytope_map:
                print('Warning: found polytope with identical centroids!')
                self.centroid_to_AHpolytope_map[hashable].append(p)
                print(self.centroid_to_AHpolytope_map[hashable])
            else:
                self.centroid_to_AHpolytope_map[hashable]=[p]
        self.centroid_to_voronoi_centroid_index = dict() #maps each centroid to its internal index in Voronoi
        self.vertex_to_voronoi_vertex_index = dict()   #maps each vertex to its internal index in Voronoi
        self.vertex_to_voronoi_centroid_index = dict()  #maps each vertex to the list of centroids using it
        self.vertex_balls = dict() #maps each vertex to a ball radius around it
        self.parse_voronoi_diagram()

        #build kd-tree for centroids
        self.centroid_tree = build_centroid_kd_tree(self.AHpolytopes)
        self.build_sphere_around_vertices()

    def parse_voronoi_diagram(self):
        for centroid_index, centroid in enumerate(self.centroid_voronoi.points):
            self.centroid_to_voronoi_centroid_index[str(centroid)] = centroid_index
        for vertex_index, vertex in enumerate(self.centroid_voronoi.vertices):
            self.vertex_to_voronoi_vertex_index[str(vertex)] = vertex_index
        for centroid in self.centroid_voronoi.points:
            vertex_indices = self.get_voronoi_vertex_indices_of_centroid(centroid)
            for vertex_index in np.atleast_1d(vertex_indices):
                vertex = self.centroid_voronoi.vertices[vertex_index]
                hashable = str(vertex)
                if hashable not in self.vertex_to_voronoi_centroid_index:
                    self.vertex_to_voronoi_centroid_index[hashable] = deque()
                self.vertex_to_voronoi_centroid_index[hashable].append(vertex_index)
        return

    def get_voronoi_vertex_indices_of_centroid(self, centroid):
        assert(str(centroid) in self.centroid_to_voronoi_centroid_index)
        region_index = self.centroid_voronoi.point_region[self.centroid_to_voronoi_centroid_index[str(centroid)]]
        vertex_indices = self.centroid_voronoi.regions[region_index]
        return vertex_indices

    def build_sphere_around_vertices(self):
        for vertex in self.centroid_voronoi.vertices:
            vertex_to_centroid_distance = np.linalg.norm(vertex-self.vertex_to_voronoi_centroid_index[str(vertex)][0])
            #construct ball around the vertex
            self.vertex_balls[str(vertex)] = vertex_to_centroid_distance
        return

    def build_cell_AHpolytope_map(self):
        for AHpolytope in self.AHpolytopes:
            #check each vertex and see if it is contained by the AHpolytope
            #FIXME
            for vertex in self.centroid_voronoi.points:
                if AHpolytope.is_inside(vertex):
                    associated_centroid_ids = self.vertex_to_voronoi_centroid_index[str(vertex)]
                    for centroid_id in associated_centroid_ids:
                        centroid = self.centroid_voronoi.points[centroid_id]
                        self.centroid_to_AHpolytope_map[str(centroid)].append(AHpolytope)

    def find_closest_AHpolytopes(self, query_point, k_closest = 1):
        #find the closest centroid
        d,i = self.centroid_tree.query(query_point)
        closest_centroid = self.centroid_tree.data[i]
        print(self.centroid_to_AHpolytope_map.keys())
        closest_AHpolytope_candidates = self.centroid_to_AHpolytope_map[str(closest_centroid)]
        #check the AHpolytopes for the closest
        sorted_AHpolytopes = sorted(closest_AHpolytope_candidates, key = lambda p: distance_point(p, query_point))
        return sorted_AHpolytopes[0: min(k_closest, len(sorted_AHpolytopes))]