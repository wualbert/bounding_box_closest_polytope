import numpy as np

class AABB:
    def __init__(self, vertices):
        '''
        Creates an axis-aligned bounding box from two diagonal vertices
        :param vertices: a list of defining vertices with shape (2, dimensions)
        '''
        try:
            assert(len(vertices[0]) == len(vertices[1]))
        except AssertionError:
            print('Mismatched vertex dimensions')
            return
        self.dimension = len(vertices[0])
        self.u = np.asarray(vertices[0])
        self.v = np.asarray(vertices[1])

        for d in range(self.dimension):
            if vertices[0][d]>vertices[1][d]:
                self.v[d], self.u[d] = vertices[0][d], vertices[1][d]

    def __repr__(self):
        return "R^%d AABB with vertices "%(self.dimension) + np.array2string(self.u) +\
               ","+np.array2string(self.v)

    def overlaps(self, b2):
        '''
        U: lower corner. V: upper corner
        :param b2: box to compare to
        :return:
        '''
        u1_leq_v2 = np.less_equal(self.u,b2.v)
        u2_leq_v1 = np.less_equal(b2.u, self.v)
        return u1_leq_v2.all() and u2_leq_v1.all()


def overlaps(a,b):
    return a.overlaps(b)