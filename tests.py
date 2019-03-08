import numpy as np
import unittest
from box import AABB

class AABBConstructionTestCase(unittest.TestCase):
    def test_dimension(self):
        x = np.random.rand(2,5)
        box = AABB([x[0,:], x[1,:]])
        self.assertEqual(box.dimension,5)

    def test_vertices(self):
        x = [(3,2,6),(1,5,5)]
        box = AABB(x)
        y = [(1,2,5),(3,5,6)]
        answer = AABB(y)
        self.assertEqual(box.dimension,3)
        self.assertEqual(box.u.any(),answer.u.any())
        self.assertEqual(box.v.any(),answer.v.any())


class AABBCollisionTestCase(unittest.TestCase):
    def test_collision_vertex_overlap(self):
        x = [(3,5,6),(1,5,5)]
        box1 = AABB(x)
        y = [(1,2,5),(3,5,6)]
        box2 = AABB(y)
        self.assertTrue(box1.overlaps(box2))

    def test_collision_contain(self):
        x = [(0,0),(5,5)]
        box1 = AABB(x)
        y = [(1,2),(3,4)]
        box2 = AABB(y)
        self.assertTrue(box1.overlaps(box2))

    def test_collision_overlap(self):
        x = [(0,0),(5,5)]
        box1 = AABB(x)
        y = [(1,2),(9,7)]
        box2 = AABB(y)
        self.assertTrue(box1.overlaps(box2))

    def test_collision_no_overlap(self):
        x = [(5,0),(0,5)]
        box1 = AABB(x)
        y = [(9,10),(-9,6)]
        box2 = AABB(y)
        self.assertFalse(box1.overlaps(box2))

class BoxNodeTestCase(unittest.TestCase):
    def 

if __name__=='__main__':
    unittest.main()