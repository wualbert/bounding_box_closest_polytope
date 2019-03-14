# -*- coding: utf-8 -*-
'''

@author: wualbert
'''

import random
import unittest

from lib.zonotope_tree import *
from visualization.visualize import *
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ

class CentroidKDTreeTestCase(unittest.TestCase):
    def test_kd_construction(self):
        pass
    def test_query(self):
        pass

class ZonotopeTreeTestCase(unittest.TestCase):
    def test_tree_construction(self):
        G_l = np.array([[1, 0, 0, 3], [0, 1, 2, -1]]) * 0.8
        G_r = np.array([[1, 0, 1, 1, 2, -2], [0, 1, 1, -1, 5, 2]]) * 1
        x_l = np.array([0, 1]).reshape(2, 1)
        x_r = np.array([5, 0]).reshape(2, 1)
        zono_l = zonotope(x_l, G_l)
        zono_r = zonotope(x_r, G_r)
        zt = ZonotopeTree([zono_l,zono_r])
        query_point = [0,-5]
        closest_zonotope = zt.find_closest_zonotopes(np.asarray(query_point))
        # print(closest_zonotope)
        fig, ax = visZ([zono_r,zono_l], title="", alpha=0.2)
        plt.scatter(query_point[0],query_point[1])
        fig, ax = visZ(closest_zonotope, title="",fig=fig,ax=ax,alpha=0.75)
        fig, ax = visualize_box_nodes(zt.box_nodes,fig=fig,ax=ax,alpha =0.4)
        print('Closest Zonotope: ', closest_zonotope)
        plt.show()


