from bounding_box.zonotope_tree import PolytopeTree

def polytree_to_zonotope_tree(polytree):
    zonotopes = []
    for s in polytree.states:
        zonotopes.append(s.p)
    # print(zonotopes)
    zt = PolytopeTree(zonotopes)
    return zt