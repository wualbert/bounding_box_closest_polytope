import numpy as np
from box import AABB

class BoxNode:
    def __init__(self, box, q, left_child=None, right_child=None, parent=None):
        self.left_child = left_child    #left child node
        self.right_child = right_child  #right child node
        self.parent = parent    #parent node
        self.q = q  #key dimension for sorting
        self.box = box
        if self.left_child:
            self.min_u =
        self.max_u =
        self.min_v =
        self.max_v =
        self.q =