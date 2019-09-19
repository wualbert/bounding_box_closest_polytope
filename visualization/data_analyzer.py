import numpy as np
import os

dir = os.path.dirname(os.path.realpath(__file__))+'/../tests/test_random_zonotope_dim20190918_20-00-52'

voronoi_precomputation_times_avg = np.load(dir+'/voronoi_precomputation_times_avg.npy')
voronoi_precomputation_times_std = np.load(dir+'/voronoi_precomputation_times_std.npy')
voronoi_query_times_avg = np.load(dir+'/voronoi_query_times_avg.npy')
voronoi_query_times_std = np.load(dir+'/voronoi_query_times_std.npy')
voronoi_query_reduction_percentages_avg = np.load(dir+'/voronoi_query_reduction_percentages_avg.npy')
voronoi_query_reduction_percentages_std = np.load(dir+'/voronoi_query_reduction_percentages_std.npy')

aabb_precomputation_times_avg = np.load(dir+'/aabb_precomputation_times_avg.npy')
aabb_precomputation_times_std = np.load(dir+'/aabb_precomputation_times_std.npy')
aabb_query_times_avg = np.load(dir+'/aabb_query_times_avg.npy')
aabb_query_times_std = np.load(dir+'/aabb_query_times_std.npy')
aabb_query_reduction_percentages_avg = np.load(dir+'/aabb_query_reduction_percentages_avg.npy')
aabb_query_reduction_percentages_std = np.load(dir+'/aabb_query_reduction_percentages_std.npy')

params = np.load(dir+'/params.npy')
print(params)


# np.save('test_random_zonotope_dim' + experiment_name + '/voronoi_precomputation_times_std',
#         voronoi_precomputation_times_std)
# np.save('test_random_zonotope_dim' + experiment_name + '/voronoi_query_times_avg', voronoi_query_times_avg)
# np.save('test_random_zonotope_dim' + experiment_name + '/voronoi_query_times_std', voronoi_query_times_std)
# np.save('test_random_zonotope_dim' + experiment_name + '/voronoi_query_reduction_percentages_avg',
#         voronoi_query_reduction_percentages_avg)
# np.save('test_random_zonotope_dim' + experiment_name + '/voronoi_query_reduction_percentages_std',
#         voronoi_query_reduction_percentages_std)
#
# np.save('test_random_zonotope_dim' + experiment_name + '/aabb_precomputation_times_avg', aabb_precomputation_times_avg)
# np.save('test_random_zonotope_dim' + experiment_name + '/aabb_precomputation_times_std', aabb_precomputation_times_std)
# np.save('test_random_zonotope_dim' + experiment_name + '/aabb_query_times_avg', aabb_query_times_avg)
# np.save('test_random_zonotope_dim' + experiment_name + '/aabb_query_times_std', aabb_query_times_std)
# np.save('test_random_zonotope_dim' + experiment_name + '/aabb_query_reduction_percentages_avg',
#         aabb_query_reduction_percentages_avg)
# np.save('test_random_zonotope_dim' + experiment_name + '/aabb_query_reduction_percentages_std',
#         aabb_query_reduction_percentages_std)
# params = np.array([['dim', np.atleast_1d(dims)], ['count', np.atleast_1d(count)],
#                    ['construction_repeats', np.atleast_1d(construction_repeats)], \
#                    ['queries', np.atleast_1d(queries)], ['seed', np.atleast_1d(seed)]])
# np.save('test_random_zonotope_dim' + experiment_name + '/params', params)