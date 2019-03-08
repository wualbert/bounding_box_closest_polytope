import numpy as np
from box import AABB

class BoxNode:
    def __init__(self, q, uq, left_child=None, right_child=None, parent=None):
        self.left_child = left_child    #left child node
        self.right_child = right_child  #right child node
        self.parent = parent    #parent node
        self.box = AABB([np.minimum(self.left_child.box.u,self.right_child.box.u),\
                        np.maximum(self.left_child.box.v, self.right_child.box.v)])
        # TODO: find best axis for sorting
        self.q = q  #key dimension for sorting
        self.uq = uq #key for sorting
        self.update_for_children()

    def update_for_children(self):
        if self.left_child:
            self.left_child.box.u = np.minimum(np.minimum(self.left_child.left_child_min_u, \
                                                          self.left_child.right_child_min_u,),\
                                                          self.left_child.box.u)
            self.left_child.box.v = np.maximum(np.maximum(self.left_child.left_child_max_v, \
                                                          self.left_child.right_child_max_v,),\
                                                          self.left_child.box.v)

        if self.right_child:
            self.right_child.box.u = np.minimum(np.minimum(self.right_child.left_child_min_u, \
                                                          self.right_child.right_child_min_u,),\
                                                          self.right_child.box.u)
            self.right_child.box.v = np.maximum(np.maximum(self.right_child.left_child_max_v, \
                                                          self.right_child.right_child_max_v,),\
                                                          self.right_child.box.v)
    def set_parent(self, parent):
        self.parent = parent

    def set_left_child(self, left_child):
        self.left_child = left_child
        self.update_for_children()

    def set_right_child(self, right_child):
        self.right_child = right_child
        self.update_for_children()

    def in_this_box(self,test_box):
        return self.box.overlaps(test_box)

    def evaluate_node(self, test_box, overlapping_box_list):
        if self.in_this_box(test_box) and \
                (self.left_child is None) and (self.right_child is None):   #leaf branch
            overlapping_box_list.append(self.box)

        if self.left_child:     #exists a left child
            #check whether to evaluate left branch
            vi_geq_l_umin = np.invert(np.less(test_box.v, self.left_child.box.u))
            ui_leq_l_vmax = np.invert(np.greater(test_box.u, self.left_child.box.v))
            if vi_geq_l_umin.all() and ui_leq_l_vmax.all():
                self.left_child.evaluate_node(test_box,overlapping_box_list)

        if self.right_child:    #exists a right child
            if test_box.v[self.q] >= self.uq:
                #check whether to evaluate right branch
                vi_geq_r_umin = np.invert(np.less(test_box.v, self.right_child.box.u))
                ui_leq_r_vmax = np.invert(np.greater(test_box.u, self.right_child.box.v))
                if vi_geq_r_umin.all() and ui_leq_r_vmax.all():
                    self.right_child.evaluate_node(test_box,overlapping_box_list)

def find_median(box_nodes, q):
    '''
    Given a list of BoxNodes and the axis of interest q, find the median that splits the BoxNodes evenly
    :param box_nodes: a list of BoxNodes
    :param q: the dimension to perform binary splitting
    :return: the median of the box_nodes' index
    '''
    uqs = np.zeros([len(box_nodes),1])
    for i,bn in enumerate(box_nodes):
        uqs[i] = bn.box.u[q]
    return np.median(uqs)

def split_by_value(box_nodes, q, m):
    '''
    Given a list of BoxNodes, the dimension to perform binary splitting, and the splitting key, split the nodes
    :param box_nodes: list of BoxNodes
    :param q: the dimension to perform binary splitting
    :param m: split index
    :return: 2 tuple of the left and right BoxNode lists
    '''
    box_node_m = box_nodes[0]
    box_nodes_l = []
    box_nodes_r = []
    for i, bn in enumerate(box_nodes):
        if bn.box.u[q]<=m:
            box_nodes_l.append(bn)
        else:
            box_nodes_r.append(bn)
        if abs(bn.box.u[q]-m)<abs(box_node_m.box.u[q]-m):
            box_node_m = bn
    return (box_nodes_l,box_nodes_r)

def binary_split(box_nodes, q, dim, parent=None, recurse_q=None):
    '''
    Binary split a list of BoxNodes
    :param box_nodes: list of BoxNodes
    :param q: the dimension to perform binary splitting
    :param parent: the parent node of box_nodes. For preventing infinite recursions
    :return: node_m, the "root" node
    '''
    # Termination condition
    if len(box_nodes)<=1:
        # at a leaf
        return box_nodes[0]
    q = np.mod(q,dim)
    m = find_median(box_nodes,q)
    box_nodes_l, box_nodes_r = split_by_value(box_nodes, q, m)
    if abs(len(box_nodes_l)-len(box_nodes_r))>1:    #unbalanced tree
        if q == np.mod(recurse_q-1):
            #FIXME: Find a better way to break ties
            pass
        else:
            return binary_split(box_nodes,q+1,dim,recurse_q=q)
    this_node = BoxNode(q, m, parent=parent)
    left_child_node = binary_split(box_nodes_l,q+1, this_node)
    right_child_node = binary_split(box_nodes_l,q+1, this_node)
    this_node.set_left_child(left_child_node)
    this_node.set_right_child(right_child_node)
    return this_node