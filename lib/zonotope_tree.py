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
        self.box_node_to_zonotope_map = dict()
        for z in self.zonotopes:
            box = zonotope_to_box(z)
            bn = BoxNode(box)
            self.box_nodes.append(bn)
            if box in self.box_node_to_zonotope_map.keys():
                self.box_node_to_zonotope_map[box].append(z)
            else:
                self.box_node_to_zonotope_map[box] = [z]
        self.root = binary_split(self.box_nodes)
        # Create kd-tree data structure from zonotopes
        self.centroid_tree = build_centroid_kd_tree(self.zonotopes)

    def find_closest_zonotopes(self,query_point):
        #find closest centroid
        _x, ind = self.centroid_tree.query(query_point)
        closest_centroid = self.centroid_tree.data[ind]
        #calculate distance from closest centroid
        edge_length = 2*np.linalg.norm(closest_centroid-query_point)
        #create query box
        query_box = AABB_centroid_edge(query_point,edge_length)
        #find candidate box nodes
        candidate_boxes = []
        self.root.evaluate_node(query_box,candidate_boxes)
        #map back to zonotopes
        closest_zonotopes = []
        closest_distance = np.inf
        for cb in candidate_boxes:
            z_list = self.box_node_to_zonotope_map[cb]
            for z in z_list:
                if len(query_point.shape)==1:
                    query_point = np.reshape(query_point,(2,1))
                candidate_d = zonotope_distance_point(z, query_point)
                if closest_distance>candidate_d:
                    closest_distance = candidate_d
                    closest_zonotopes = [z]
                elif closest_distance==candidate_d:
                    closest_zonotopes.append(z)
        return closest_zonotopes


def build_centroid_kd_tree(zonotopes):
    n = len(zonotopes)
    k = zonotopes[0].x.shape[0]
    centroids = np.zeros((n,k))
    for i, z in enumerate(zonotopes):
        centroids[i,:] = zonotopes[i].x[:,0]
    return KDTree(centroids)