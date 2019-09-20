import numpy as np
import os
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib
from visualization.visualize import *

matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams.update({'font.size': 15})

# Load
dir = '/Users/albertwu/Google Drive/MIT/RobotLocomotion/Closest Polytope/ACC2020/Results/test_random_zonotope_dim20190919_01-56-56'#os.path.dirname(os.path.realpath(__file__))+'/../tests/test_random_zonotope_dim20190919_01-56-56'

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

for p in params:
    if p[0] == 'count':
        counts = p[1]
    if p[0] == 'dim':
        dims = p[1]
# Plot
fig_index = 0
plt.figure(fig_index)
fig_index += 1
plt.subplot(111)
plt.errorbar(dims, voronoi_precomputation_times_avg, voronoi_precomputation_times_std, marker='.', color='b',
             ecolor='b', elinewidth=0.3,
             capsize=2, linewidth=0.5, markersize=7)
plt.errorbar(dims, aabb_precomputation_times_avg, aabb_precomputation_times_std, marker='.', color='r', ecolor='r',
             elinewidth=0.3,
             capsize=2, linewidth=0.5, markersize=7)
plt.legend(['Centroid Dist.', 'AABB'])
plt.xlabel('State Dimension')
plt.ylabel('Precomputation Time (s)')
plt.title('Closest Zonotope Precomputation Time with %d Zonotopes' % counts)
#
# plt.subplot(212)
# plt.plot(dims, np.log(voronoi_precomputation_times_avg))
# plt.xlabel('$log$ State Dimension')
# plt.ylabel('$log$ Precomputation Time (s)')
# # plt.title('Voronoi Closest Zonotope Precomputation Time with %d Zonotopes' %count)
# plt.tight_layout()

plt.figure(fig_index)
fig_index += 1
plt.subplot(111)
plt.errorbar(dims, voronoi_query_times_avg, voronoi_query_times_std, marker='.', color='b', ecolor='b', elinewidth=0.3,
             capsize=2,
             linewidth=0.5, markersize=7)
plt.errorbar(dims, aabb_query_times_avg, aabb_query_times_std, marker='.', color='r', ecolor='r', elinewidth=0.3,
             capsize=2,
             linewidth=0.5, markersize=7)
plt.legend(['Centroid Dist.', 'AABB'])
plt.xlabel('State Dimension')
plt.ylabel('Query Time (s)')
plt.title('Closest Zonotope Single Query Time with %d Zonotopes' % counts)

# plt.subplot(212)
# plt.plot(np.log(dims), np.log(voronoi_query_times_avg))
# plt.xlabel('$log$ State Dimension')
# plt.ylabel('$log$ Query Time (s)')
# # plt.title('$log$ Voronoi Closest Zonotope Single Query Time with %d Zonotopes' %count)
# plt.tight_layout()

plt.figure(fig_index)
fig_index += 1
plt.subplot(111)
plt.errorbar(dims, voronoi_query_reduction_percentages_avg, voronoi_query_reduction_percentages_std, marker='.',
             color='b', ecolor='b',
             elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
plt.errorbar(dims, aabb_query_reduction_percentages_avg, aabb_query_reduction_percentages_std, marker='.', color='r',
             ecolor='r',
             elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
plt.legend(['Centroid Dist.', 'AABB'])
plt.xlabel('State Dimension')
plt.ylabel('% of Zonotopes Evaluated')
plt.title('Closest Zonotope Reduction Percentage with %d Zonotopes' % counts)

# plt.subplot(212)
# plt.plot(np.log(dims), np.log(voronoi_query_reduction_percentages_avg))
# plt.xlabel('$log$ State Dimension')
# plt.ylabel('$log$ % of Zonotopes Evaluated')
# # plt.title('$log$ Voronoi Closest Zonotope Reduction Percentage with %d Zonotopes' %count)
# plt.tight_layout()
plt.show()
