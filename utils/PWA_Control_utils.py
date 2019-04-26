from bounding_box.zonotope_tree import ZonotopeTree

def polytree_to_zonotope_tree(polytree):
    zonotopes = []
    for s in polytree.states:
        zonotopes.append(s.p)
    # print(zonotopes)
    zt = ZonotopeTree(zonotopes)
    return zt