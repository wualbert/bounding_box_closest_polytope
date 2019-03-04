import numpy as np
from box import AABB

class box_set:
    def __init__(self, boxes):
        '''

        :param boxes: list of AABB boxes
        '''
        self.boxes = boxes

        #Construct tree


    def closest_box(self, query_point):
        '''

        :param query_point: point of interest, represented by d dimensional numpy array
        :return: closest box to query point
        '''
        try:
            assert(query_point.shape[0] == self.boxes[0].dimension)
        except AssertionError:
            return None
        