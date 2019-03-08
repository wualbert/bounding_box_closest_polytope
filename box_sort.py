import numpy as np
from box import AABB

class BoxNode:
    def __init__(self, box, q, uq, left_child=None, right_child=None, parent=None):
        self.left_child = left_child    #left child node
        self.right_child = right_child  #right child node
        self.parent = parent    #parent node
        self.q = q  #key dimension for sorting
        self.uq = uq #key for sorting
        self.box = box
        self.left_child_min_u = np.inf
        self.left_child_max_v = -np.inf
        self.right_child_min_u = np.inf
        self.right_child_max_v = -np.inf
        if self.left_child:
            self.left_child_min_u = np.minimum(np.minimum(self.left_child.left_child_min_u, \
                                                          self.left_child.right_child_min_u,),\
                                                          self.left_child.box.u)
            self.left_child_max_v = np.maximum(np.maximum(self.left_child.left_child_max_v, \
                                                          self.left_child.right_child_max_v,),\
                                                          self.left_child.box.v)

        if self.right_child:
            self.right_child_min_u = np.minimum(np.minimum(self.right_child.left_child_min_u, \
                                                          self.right_child.right_child_min_u,),\
                                                          self.right_child.box.u)
            self.right_child_max_v = np.maximum(np.maximum(self.right_child.left_child_max_v, \
                                                          self.right_child.right_child_max_v,),\
                                                          self.right_child.box.v)

    def in_this_box(self,test_box):
        return self.box.overlaps(test_box)

    def evaluate_node(self, test_box, overlapping_box_list):
        if self.in_this_box(test_box):
            overlapping_box_list.append(self.box)

        if self.left_child:     #exists a left child
            #check whether to evaluate left branch
            vi_geq_l_umin = np.invert(np.less(test_box.v, self.left_child_min_u))
            ui_leq_l_vmax = np.invert(np.greater(test_box.u, self.left_child_max_v))
            if vi_geq_l_umin.all() and ui_leq_l_vmax.all():
                self.left_child.evaluate_node(test_box,overlapping_box_list)

        if self.right_child:    #exists a right child
            if test_box.v[self.q] >= self.uq:
                #check whether to evaluate right branch
                vi_geq_r_umin = np.invert(np.less(test_box.v, self.right_child_min_u))
                ui_leq_r_vmax = np.invert(np.greater(test_box.u, self.right_child_max_v))
                if vi_geq_r_umin.all() and ui_leq_r_vmax.all():
                    self.right_child.evaluate_node(test_box,overlapping_box_list)

