import numpy as np

class Polytope:
    '''
    A polytope is an object. H-rep is used: {x | H x \le h}
    Code from Sadra Sadraddini
    '''

    def __init__(self, H, h, dimensions="full"):
        self.H = H
        self.h = h
        self.dimensions = dimensions

    def __repr__(self):
        return "polytope in R^%d" % self.H.shape[1]


class ClosestPolytopeStructure:
    def __init__(self):

        self.centroid_kd_tree = None

    def construct_centroid_kd_tree(self):

    def contruct_AABB(self):
        pass
