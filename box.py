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


def point_to_box_distance(point, box):
    out_range_dim = []
    for dim in range(box.dimension):
        if box.u[dim] < point[dim] and box.v[dim] > point[dim]:
            pass
        else:
            out_range_dim.append(min(abs(box.u[dim]-point[dim]), abs(box.v[dim]-point[dim])))
    return np.linalg.norm(out_range_dim)


def box_to_box_distance(query_box, box):
    out_range_dim = []
    for dim in range(box.dimension):
        if (box.u[dim] < query_box.u[dim] and box.v[dim] > query_box.u[dim]) or \
            (box.u[dim] < query_box.v[dim] and box.v[dim] > query_box.v[dim]) or \
            (query_box.u[dim] < box.u[dim] and query_box.v[dim] > box.u[dim]) or \
            (query_box.u[dim] < box.v[dim] and query_box.v[dim] > box.v[dim]):
            pass
        else:
            out_range_dim.append(min(min(abs(box.u[dim]-query_box.u[dim]), abs(box.v[dim]-query_box.u[dim])),
                                 min(abs(box.u[dim]-query_box.v[dim]), abs(box.v[dim]-query_box.v[dim]))))
    return np.linalg.norm(out_range_dim)
