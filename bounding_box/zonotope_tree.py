# -*- coding: utf-8 -*-
'''

@author: wualbert
'''
from box_tree import *
from box import *
from pypolycontain.lib.zonotope import zonotope_distance_point, distance_point
from utils.utils import build_centroid_kd_tree

class PolytopeTree:
    def __init__(self, polytopes):
        self.polytopes = polytopes
        # Create box data structure from zonotopes
        self.type = self.polytopes[0].type
        self.box_nodes = []
        self.centroid_tree = build_centroid_kd_tree(self.polytopes)
        self.centroid_to_zonotope_map = dict()

        for z in self.polytopes:
            if self.type == 'zonotope':
                box = zonotope_to_box(z)
            elif self.type == 'AH_polytope' or 'H-polytope':
                box = AH_polytope_to_box(z)
            else:
                raise NotImplementedError
            bn = BoxNode(box)
            self.box_nodes.append(bn)
            if z.x.tostring() in self.centroid_to_zonotope_map:
                assert(False)
                self.centroid_to_zonotope_map[z.x.tostring()].append(z)
            else:
                self.centroid_to_zonotope_map[z.x.tostring()]=[z]
        assert(len(self.box_nodes) == len(self.polytopes))
        self.root = binary_split(self.box_nodes)
        # Create kd-tree data structure from zonotopes

    def find_closest_zonotopes(self,query_point, return_intermediate_info=False):
        #find closest centroid
        try:
            query_point.shape[1]
            #FIXME: Choose between d*1 (2D) or d (1D) represntation of query_point
        except:
            # raise ValueError('Query point should be d*1 numpy array')
            query_point=query_point.reshape((-1,1))
        _x, ind = self.centroid_tree.query(np.ndarray.flatten(query_point))
        closest_centroid = self.centroid_tree.data[ind]
        # print('Closest centroid', closest_centroid)

        #Use dist(centroid, query) as upper bound
        vector_diff = np.subtract(np.ndarray.flatten(closest_centroid),\
                                  np.ndarray.flatten(query_point))
        # edge_length = 2*np.linalg.norm(vector_diff)

        #Use dist(polytope, query) as upper bound
        centroid_zonotopes = self.centroid_to_zonotope_map[closest_centroid.tostring()]
        edge_length = np.inf
        edge_zonotope = None
        for cz in centroid_zonotopes:
            zd = zonotope_distance_point(cz,query_point)
            if edge_length > zd:
                edge_length=zd
                edge_zonotope=cz

        #create query box
        query_box = AABB_centroid_edge(query_point,2*edge_length)
        #find candidate box nodes
        candidate_boxes = []
        self.root.evaluate_node(query_box,candidate_boxes)
        # print('Evaluating %d zonotopes') %len(candidate_boxes)
        #map back to zonotopes
        closest_zonotopes = []
        closest_distance = np.inf
        if candidate_boxes is None:
            # This should never happen
            raise ValueError('No closest zonotope found!')
            # When a heuristic less than centroid distance is used,
            # a candidate box does not necessarily exist. In this case,
            # use the zonotope from which the heuristic is generated.
            # '''
            #
            # closest_zonotopes = edge_zonotope
            # closest_distance = edge_length
            # return closest_zonotopes, candidate_boxes, query_box
        else:
            for cb in candidate_boxes:
                closest_zonotopes.append(cb.zonotope)
            #find the closest zonotope
            best_polytope = None
            best_distance = np.inf
            for p in closest_zonotopes:
                dist = zonotope_distance_point(p, query_point)
                if best_distance > dist:
                    best_distance = dist
                    best_polytope = p
            if return_intermediate_info:
                return best_polytope, best_distance, closest_zonotopes
            return best_polytope