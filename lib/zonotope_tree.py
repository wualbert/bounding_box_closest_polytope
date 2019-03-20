# -*- coding: utf-8 -*-
'''

@author: wualbert
'''
from box_tree import *
from box import *
from scipy.spatial import KDTree
from pypolycontain.lib.zonotope import zonotope_distance_point

class ZonotopeTree:
    def __init__(self,zonotopes):
        self.zonotopes = zonotopes
        # Create box data structure from zonotopes
        self.box_nodes = []
        self.centroid_tree = build_centroid_kd_tree(self.zonotopes)
        self.centroid_to_zonotope_map = dict()

        for z in self.zonotopes:
            box = zonotope_to_box(z)
            bn = BoxNode(box)
            self.box_nodes.append(bn)
            if z.x.tostring() in self.centroid_to_zonotope_map:
                assert(False)
                self.centroid_to_zonotope_map[z.x.tostring()].append(z)
            else:
                self.centroid_to_zonotope_map[z.x.tostring()]=[z]
        assert(len(self.box_nodes)==len(self.zonotopes))
        self.root = binary_split(self.box_nodes)
        # Create kd-tree data structure from zonotopes

    def find_closest_zonotopes(self,query_point):
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
            '''
            When a heuristic less than centroid distance is used,
            a candidate box does not necessarily exist. In this case,
            use the zonotope from which the heuristic is generated.
            '''
            closest_zonotopes = edge_zonotope
            closest_distance = edge_length
            return closest_zonotopes, candidate_boxes, query_box
        else:
            for cb in candidate_boxes:
                candidate_d = zonotope_distance_point(cb.zonotope, query_point)
                if closest_distance>candidate_d:
                    closest_distance = candidate_d
                    closest_zonotopes = [cb.zonotope]
                elif closest_distance==candidate_d:
                    closest_zonotopes.append(cb.zonotope)
            return closest_zonotopes, candidate_boxes, query_box


def build_centroid_kd_tree(zonotopes):
    n = len(zonotopes)
    k = zonotopes[0].x.shape[0]
    centroids = np.zeros((n,k))
    for i, z in enumerate(zonotopes):
        centroids[i,:] = zonotopes[i].x[:,0]
    return KDTree(centroids)