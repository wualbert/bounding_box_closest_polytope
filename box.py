import numpy as np

class AABB:
    def __init__(self, vertices):
        '''
        Creates an axis-aligned bounding box from two diagonal vertices
        :param vertices: a numpy array of defining vertices with shape (2, dimensions)
        '''
        self.dimension = vertices.shape[1]
        self.vertices = vertices

    def __repr__(self):
        '''

        :return:
        '''
        return "R^%d AABB with vertices "%(self.dimension) + str(self.vertices)