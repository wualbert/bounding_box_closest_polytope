import random
import unittest

from lib.box import *
from lib.box_tree import *
from visualization.visualize import *


class BoxToBoxDistanceTestCase(unittest.TestCase):
    def test_distance_to_box(self):
        x = [(0,0,0),(3,3,3)]
        box1 = AABB(x)
        q1 = AABB([(1,1,1),(3,3,3)])
        self.assertTrue(box_to_box_distance(q1, box1)==0)
        q2 = AABB([(8,8,8),(3,3,3)])
        self.assertTrue(box_to_box_distance(q2,box1)==0)
        q3 = AABB([(2,2,2),(8,8,8)])
        self.assertTrue(box_to_box_distance(q3, box1)==0)
        q4 = AABB([(8,8,8),(4,4,4)])
        self.assertTrue(box_to_box_distance(q4, box1) == 3**0.5)
        q5 = AABB([(4,0,0),(7,3,3)])
        self.assertTrue(box_to_box_distance(q5, box1) == 1)

class BoxNodeTestCase(unittest.TestCase):
    def test_construct_box_node(self):
        x = [(1,1),(3,3)]
        box1 = AABB(x)
        y = [(2,2),(4,4)]
        box2 = AABB(y)
        z = [(3,3),(5,5)]
        box3 = AABB(z)

        bn1 = BoxNode(box1)
        self.assertTrue(bn1.in_this_box(box2))

class BoxTreeTestCase(unittest.TestCase):
    def test_construct_box_tree(self):
        box_list = []
        box_node_list = []
        for i in range(10):
            xs = random.sample(range(100), 2)
            ys = random.sample(range(100),2)
            u = (xs[0],ys[0])
            v = (xs[1],ys[1])
            box = AABB([u,v])
            box_list.append(box)
            box_node_list.append(BoxNode(box))
        root = binary_split(box_node_list)
        # print('root',root)
        # print(box_node_list)

    def test_closest_box(self):
        box_list = []
        box_node_list = []
        for i in range(10):
            xs = random.sample(range(100), 2)
            ys = random.sample(range(100),2)
            u = (xs[0],ys[0])
            v = (xs[1],ys[1])
            box = AABB([u,v])
            box_list.append(box)
            box_node_list.append(BoxNode(box))

        overlapping_box_list = []
        closest_distance = np.inf
        root = binary_split(box_node_list)
        xs = random.sample(range(100), 2)
        ys = random.sample(range(100), 2)
        u = (xs[0], ys[0])
        v = (xs[1], ys[1])
        test_box = AABB([u, v])
        print('test lib: ', test_box)
        root.evaluate_node(test_box,overlapping_box_list)
        print('overlaps with', overlapping_box_list)
        for box in box_list:
            #FIXME: slow implementation
            if box in overlapping_box_list:
                # print(lib)
                # print(box_to_box_distance(test_box, lib))
                self.assertTrue(box_to_box_distance(test_box,box)==0)
            else:
                self.assertTrue(box_to_box_distance(test_box, box)>0)

if __name__=='__main__':
    unittest.main()