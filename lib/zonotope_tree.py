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
        for z in self.zonotopes:
            box = zonotope_to_box(z)
            bn = BoxNode(box)
            self.box_nodes.append(bn)
        assert(len(self.box_nodes)==len(self.zonotopes))
        self.root = binary_split(self.box_nodes)
        # Create kd-tree data structure from zonotopes
        self.centroid_tree = build_centroid_kd_tree(self.zonotopes)

    def find_closest_zonotopes(self,query_point):
        #find closest centroid
        try:
            query_point.shape[1]
            #FIXME: Choose between d*1 (2D) or d (1D) represntation of query_point
        except:
            raise ValueError('Query point should be d*1 numpy array')
        _x, ind = self.centroid_tree.query(np.ndarray.flatten(query_point))
        closest_centroid = self.centroid_tree.data[ind]
        # print('Closest centroid', closest_centroid)
        vector_diff = np.subtract(np.ndarray.flatten(closest_centroid),\
                                  np.ndarray.flatten(query_point))
        # print(vector_diff)
        edge_length = 2*np.linalg.norm(vector_diff)
        # print('Edge_length', edge_length)
        #create query box
        query_box = AABB_centroid_edge(query_point,edge_length)
        #find candidate box nodes
        candidate_boxes = []
        self.root.evaluate_node(query_box,candidate_boxes)
        # print(query_box)
        #map back to zonotopes
        closest_zonotopes = []
        closest_distance = np.inf
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