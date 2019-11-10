# -*- coding: utf-8 -*-
'''

@author: wualbert
'''
from closest_polytope_algorithms.bounding_box.box_tree import *
from closest_polytope_algorithms.bounding_box.box import *
from pypolycontain.lib.operations import distance_point_polytope
try:
    from closest_polytope_algorithms.utils.utils import build_key_point_kd_tree
except:
    from closest_polytope_algorithms.utils.utils import build_key_point_kd_tree
from rtree import index

class PolytopeTree:
    def __init__(self, polytopes, key_vertex_count = 0):
        '''
        Updated implementation using rtree
        :param polytopes:
        '''
        self.polytopes = polytopes
        self.key_vertex_count = key_vertex_count
        # Create box data structure from zonotopes
        self.type = self.polytopes[0].type
        # Initialize rtree structure
        self.rtree_p = index.Property()
        self.rtree_p.dimension = to_AH_polytope(self.polytopes[0]).t.shape[0]
        print(('PolytopeTree dimension is %d-D' % self.rtree_p.dimension))
        self.idx = index.Index(properties=self.rtree_p)
        self.index_to_polytope_map = {}
        for z in self.polytopes:
            if self.type == 'zonotope':
                lu = zonotope_to_box(z)
            elif self.type == 'AH_polytope' or 'H-polytope':
                lu = AH_polytope_to_box(z)
            else:
                raise NotImplementedError
            # assert(hash(z) not in self.index_to_polytope_map)
            #FIXME
            if hash(z) not in self.index_to_polytope_map:
                self.idx.insert(hash(z), lu)
                self.index_to_polytope_map[hash(z)] = z

        # build key point tree for query box size guess
        self.key_point_tree, self.key_point_to_zonotope_map = build_key_point_kd_tree(self.polytopes, self.key_vertex_count)

    def insert(self, new_polytopes):
        '''
        Inserts a new polytope to the tree structure
        :param new_polytope:
        :return:
        '''
        # insert into rtree
        for new_polytope in new_polytopes:
            if new_polytope.type == 'zonotope':
                lu = zonotope_to_box(new_polytope)
            elif new_polytope.type == 'AH_polytope' or 'H-polytope':
                lu = AH_polytope_to_box(new_polytope)
            else:
                raise NotImplementedError
            self.idx.insert(hash(new_polytope), lu)
            # assert (hash(new_polytope) not in self.index_to_polytope_map)
            self.index_to_polytope_map[hash(new_polytope)] = new_polytope
        if isinstance(self.polytopes, np.ndarray):
            self.polytopes = np.concatenate((self.polytopes,np.array(new_polytopes)))
        else:
            self.polytopes.append(new_polytope)
            # insert into kdtree
        # FIXME: Rebuilding a kDtree should not be necessary
        self.key_point_tree, self.key_point_to_zonotope_map = build_key_point_kd_tree(self.polytopes, self.key_vertex_count)

    def find_closest_polytopes(self, query_point, return_intermediate_info=False):
        #find closest centroid
        # try:
        #     query_point.shape[1]
        #     #FIXME: Choose between d*1 (2D) or d (1D) representation of query_point
        # except:
        #     # raise ValueError('Query point should be d*1 numpy array')
        #     query_point=query_point.reshape((-1,1))

        # Construct centroid box
        _x, ind = self.key_point_tree.query(np.ndarray.flatten(query_point))
        closest_centroid = self.key_point_tree.data[ind]
        # print('Closest centroid', closest_centroid)

        #Use dist(centroid, query) as upper bound
        vector_diff = np.subtract(np.ndarray.flatten(closest_centroid),\
                                  np.ndarray.flatten(query_point))
        # pivot_distance = 2*np.linalg.norm(vector_diff)

        #Use dist(polytope, query) as upper bound
        evaluated_zonotopes = []
        centroid_zonotopes = self.key_point_to_zonotope_map[closest_centroid.tostring()]
        best_distance = np.inf
        # best_inf_distance = np.inf
        best_polytope = None
        dist_to_query = {}
        # inf_dist_to_query = {}

        assert(len(centroid_zonotopes)==1)
        for cz in centroid_zonotopes:
            evaluated_zonotopes.append(cz)
            zd = distance_point_polytope(cz,query_point, ball='l2')[0]
            # zd = distance_point_polytope(cz, query_point, ball='l2')[0]
            if best_distance > zd:
                best_distance=zd
                # best_inf_distance=zd
                best_polytope=cz
                dist_to_query[cz] = best_distance
                # inf_dist_to_query[cz] = best_inf_distance

        u = query_point - best_distance
        v = query_point + best_distance
        heuristic_box_lu = np.concatenate([u, v])
        #create query box
        #find candidate box nodes
        candidate_ids = list(self.idx.intersection(heuristic_box_lu))
        # print('Evaluating %d zonotopes') %len(candidate_boxes)
        #map back to zonotopes
        if candidate_ids is None:
            # This should never happen
            raise ValueError('No closest zonotope found!')
            # When a heuristic less than centroid distance is used,
            # a candidate box does not necessarily exist. In this case,
            # use the zonotope from which the heuristic is generated.
            # '''
            #
            # evaluated_zonotopes = pivot_polytope
            # closest_distance = pivot_distance
            # return evaluated_zonotopes, candidate_boxes, query_box
        else:
            # for cb in candidate_boxes:
            #     print(cb)
            #     evaluated_zonotopes.append(cb.polytope)
            #find the closest zonotope with randomized approach]
            while(len(candidate_ids)>=1):
                if best_distance < 1e-9:
                    # point is contained by polytope, break
                    break
                sample = np.random.randint(len(candidate_ids))
                #solve linear program for the sampled polytope
                pivot_polytope = self.index_to_polytope_map[candidate_ids[sample]]
                if pivot_polytope==best_polytope:
                    #get rid of this polytope
                    candidate_ids[sample], candidate_ids[-1] = candidate_ids[-1], candidate_ids[sample]
                    candidate_ids = candidate_ids[0:-1]
                    continue
                if pivot_polytope not in dist_to_query:
                    pivot_distance = distance_point_polytope(pivot_polytope, query_point, ball="l2")[0]
                    print('pd', pivot_distance)
                    # inf_pivot_distance = distance_point_polytope(pivot_polytope, query_point)[0]
                    dist_to_query[pivot_polytope] = pivot_distance
                    # inf_dist_to_query[pivot_polytope] = inf_dist_to_query
                    if return_intermediate_info:
                        evaluated_zonotopes.append(pivot_polytope)
                else:
                    print('pp', pivot_polytope)
                    pivot_distance = dist_to_query[pivot_polytope]
                    print('pd2', pivot_distance)
                    # inf_pivot_distance = inf_dist_to_query[pivot_polytope]
                print(pivot_distance, best_distance)
                if pivot_distance>=best_distance:#fixme: >= or >?
                    #get rid of this polytope
                    candidate_ids[sample], candidate_ids[-1] = candidate_ids[-1], candidate_ids[sample]
                    candidate_ids = candidate_ids[0:-1]
                else:
                    #reconstruct AABB
                    # create query box
                    u = query_point - pivot_distance
                    v = query_point + pivot_distance
                    heuristic_box_lu = np.concatenate([u, v])
                    # find new candidates
                    candidate_ids = list(self.idx.intersection(heuristic_box_lu))
                    best_distance = pivot_distance
                    # best_inf_distance = inf_pivot_distance
                    best_polytope = pivot_polytope
            if return_intermediate_info:
                return np.atleast_1d(best_polytope), best_distance, evaluated_zonotopes, heuristic_box_lu
            return np.atleast_1d(best_polytope)
#
# class PolytopeTree_Old:
#     '''
#     Deprecated implementation. RTree is now used for the underlying data structure
#     '''
#     def __init__(self, polytopes):
#         self.polytopes = polytopes
#         # Create box data structure from zonotopes
#         self.type = self.polytopes[0].type
#         self.box_nodes = []
#         for z in self.polytopes:
#             if self.type == 'zonotope':
#                 box = zonotope_to_box(z)
#             elif self.type == 'AH_polytope' or 'H-polytope':
#                 box = AH_polytope_to_box(z, return_AABB=True)
#             else:
#                 raise NotImplementedError
#             bn = BoxNode(box)
#             self.box_nodes.append(bn)
#         assert (len(self.box_nodes) == len(self.polytopes))
#         self.key_point_tree, self.key_point_to_zonotope_map = build_key_point_kd_tree(self.polytopes)
#         self.root = binary_split(self.box_nodes)
#         # Create kd-tree data structure from zonotopes
#
#     def find_closest_zonotopes(self,query_point, return_intermediate_info=False):
#         #find closest centroid
#         try:
#             query_point.shape[1]
#             #FIXME: Choose between d*1 (2D) or d (1D) represntation of query_point
#         except:
#             # raise ValueError('Query point should be d*1 numpy array')
#             query_point=query_point.reshape((-1,1))
#         _x, ind = self.key_point_tree.query(np.ndarray.flatten(query_point))
#         closest_centroid = self.key_point_tree.data[ind]
#         # print('Closest centroid', closest_centroid)
#
#         #Use dist(centroid, query) as upper bound
#         vector_diff = np.subtract(np.ndarray.flatten(closest_centroid),\
#                                   np.ndarray.flatten(query_point))
#         # pivot_distance = 2*np.linalg.norm(vector_diff)
#
#         #Use dist(polytope, query) as upper bound
#         evaluated_zonotopes = []
#         centroid_zonotopes = self.key_point_to_zonotope_map[closest_centroid.tostring()]
#         best_distance = np.inf
#         best_polytope = None
#         for cz in centroid_zonotopes:
#             evaluated_zonotopes.append(cz)
#             zd = zonotope_distance_point(cz,query_point)[0]
#             if best_distance > zd:
#                 best_distance=zd
#                 best_polytope=cz
#
#         #create query box
#         query_box = AABB_centroid_edge(query_point,2*best_distance)
#         #find candidate box nodes
#         candidate_boxes = []
#         self.root.evaluate_node(query_box,candidate_boxes)
#         # print('Evaluating %d zonotopes') %len(candidate_boxes)
#         #map back to zonotopes
#         closest_distance = np.inf
#         if candidate_boxes is None:
#             # This should never happen
#             raise ValueError('No closest zonotope found!')
#             # When a heuristic less than centroid distance is used,
#             # a candidate box does not necessarily exist. In this case,
#             # use the zonotope from which the heuristic is generated.
#             # '''
#             #
#             # evaluated_zonotopes = pivot_polytope
#             # closest_distance = pivot_distance
#             # return evaluated_zonotopes, candidate_boxes, query_box
#         else:
#             # for cb in candidate_boxes:
#             #     print(cb)
#             #     evaluated_zonotopes.append(cb.polytope)
#             #find the closest zonotope with randomized approach
#             while(len(candidate_boxes)>1):
#                 sample = np.random.randint(len(candidate_boxes))
#                 #solve linear program for the sampled polytope
#                 pivot_polytope = candidate_boxes[sample].polytope
#                 if return_intermediate_info:
#                     evaluated_zonotopes.append(pivot_polytope)
#                 pivot_distance = distance_point_polytope(pivot_polytope, query_point, ball="l2")[0]
#                 if pivot_distance>=best_distance:#fixme: >= or >?
#                     #get rid of this polytope
#                     candidate_boxes[sample], candidate_boxes[-1] = candidate_boxes[-1], candidate_boxes[sample]
#                     candidate_boxes = candidate_boxes[0:-1]
#                 else:
#                     #reconstruct AABB
#                     # create query box
#                     query_box = AABB_centroid_edge(query_point, 2 * pivot_distance)
#                     # find candidate box nodes
#                     candidate_boxes = []
#                     self.root.evaluate_node(query_box, candidate_boxes)
#                     best_distance = pivot_distance
#                     best_polytope = pivot_polytope
#             if return_intermediate_info:
#                 return np.atleast_1d(best_polytope), best_distance, evaluated_zonotopes, query_box
#             return np.atleast_1d(best_polytope)

