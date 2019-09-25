import numpy as np
import os
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib
from visualization.visualize import *

matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams.update({'font.size': 15})

# Load
dir = '/Users/albertwu/exp/closest_polytope/tests/test_random_zonotope_count20190925_04-59-27'#os.path.dirname(os.path.realpath(__file__))+'/../tests/test_random_zonotope_dim20190919_01-56-56'

voronoi_precomputation_times_median = np.load(dir+'/voronoi_precomputation_times_median.npy')
voronoi_precomputation_times_min = np.load(dir+'/voronoi_precomputation_times_min.npy')
voronoi_precomputation_times_max = np.load(dir+'/voronoi_precomputation_times_max.npy')
voronoi_query_times_median = np.load(dir+'/voronoi_query_times_median.npy')
voronoi_query_times_min = np.load(dir+'/voronoi_query_times_min.npy')
voronoi_query_times_max = np.load(dir+'/voronoi_query_times_max.npy')
voronoi_query_reduction_percentages_median = np.load(dir+'/voronoi_query_reduction_percentages_median.npy')
voronoi_query_reduction_percentages_min = np.load(dir+'/voronoi_query_reduction_percentages_min.npy')
voronoi_query_reduction_percentages_max = np.load(dir+'/voronoi_query_reduction_percentages_max.npy')

aabb_precomputation_times_median = np.load(dir+'/aabb_precomputation_times_median.npy')
aabb_precomputation_times_min = np.load(dir+'/aabb_precomputation_times_min.npy')
aabb_precomputation_times_max = np.load(dir+'/aabb_precomputation_times_max.npy')
aabb_query_times_median = np.load(dir+'/aabb_query_times_median.npy')
aabb_query_times_min = np.load(dir+'/aabb_query_times_min.npy')
aabb_query_times_max = np.load(dir+'/aabb_query_times_max.npy')
aabb_query_reduction_percentages_median = np.load(dir+'/aabb_query_reduction_percentages_median.npy')
aabb_query_reduction_percentages_min = np.load(dir+'/aabb_query_reduction_percentages_min.npy')
aabb_query_reduction_percentages_max = np.load(dir+'/aabb_query_reduction_percentages_max.npy')


params = np.load(dir+'/params.npy')
# print(params)

for p in params:
    if p[0] == 'count':
        counts = p[1]
    if p[0] == 'dim':
        dims = p[1]

voronoi_query_count_median = np.multiply(counts, voronoi_query_reduction_percentages_median)/100
voronoi_query_count_min =np.multiply(counts, voronoi_query_reduction_percentages_min)/100
voronoi_query_count_max =np.multiply(counts, voronoi_query_reduction_percentages_max)/100

aabb_query_count_median = np.multiply(counts, aabb_query_reduction_percentages_median)/100
aabb_query_count_min = np.multiply(counts, aabb_query_reduction_percentages_min)/100
aabb_query_count_max = np.multiply(counts, aabb_query_reduction_percentages_max)/100

# Plot
fig_index = 0
plt.figure(fig_index)
fig_index += 1
plt.subplot(111)

plt.errorbar(counts, voronoi_precomputation_times_median, np.vstack([voronoi_precomputation_times_median-voronoi_precomputation_times_min, voronoi_precomputation_times_max-voronoi_precomputation_times_median]), marker='.', color='b', ecolor='b', elinewidth=0.3,
             capsize=2,
             linewidth=0.5, markersize=7)
plt.errorbar(counts, aabb_precomputation_times_median, np.vstack([aabb_precomputation_times_median-aabb_precomputation_times_min, aabb_precomputation_times_max-aabb_precomputation_times_median]), marker='.', color='r', ecolor='r', elinewidth=0.3,
             capsize=2,
             linewidth=0.5, markersize=7)

plt.legend(['Triangle Ineq.', 'AABB'])
plt.xlabel('Number of Polytopes')
plt.ylabel('Precomputation Time (s)')
plt.title('Dataset Size Scalability with Uniformly Distributed Polytopes')
plt.tight_layout()
plt.savefig('precomputation_time.png', dpi=500)
# plt.subplot(212)
# plt.plot(counts, np.log(voronoi_precomputation_times_median))
# plt.xlabel('$log$ State Number of Polytopes')
# plt.ylabel('$log$ Precomputation Time (s)')
# # plt.title('Voronoi Closest Zonotope Precomputation Time with %d Zonotopes' %count)
plt.tight_layout()

plt.figure(fig_index)
fig_index += 1
plt.subplot(111)
plt.errorbar(counts, voronoi_query_times_median, np.vstack([voronoi_query_times_median-voronoi_query_times_min, voronoi_query_times_max-voronoi_query_times_median]), marker='.', color='b', ecolor='b', elinewidth=0.3,
             capsize=2,
             linewidth=0.5, markersize=7)
plt.errorbar(counts, aabb_query_times_median, np.vstack([aabb_query_times_median-aabb_query_times_min, aabb_query_times_max-aabb_query_times_median]), marker='.', color='r', ecolor='r', elinewidth=0.3,
             capsize=2,
             linewidth=0.5, markersize=7)
plt.legend(['Triangle Ineq.', 'AABB'],loc='lower right')
plt.xlabel('Number of Polytopes')
plt.ylabel('Query Time (s)')
plt.yscale('log')
plt.title('Dataset Size Scalability with Uniformly Distributed Polytopes')

plt.tight_layout()
plt.savefig('online_uniform_count_time.png', dpi=500)
# plt.subplot(212)
# plt.plot(np.log(counts), np.log(voronoi_query_times_median))
# plt.xlabel('$log$ State Number of Polytopes')
# plt.ylabel('$log$ Query Time (s)')
# # plt.title('$log$ Voronoi Closest Zonotope Single Query Time with %d Zonotopes' %count)

plt.figure(fig_index)
fig_index += 1
plt.subplot(111)
plt.errorbar(counts, voronoi_query_reduction_percentages_median, np.vstack([voronoi_query_reduction_percentages_median-voronoi_query_reduction_percentages_min, voronoi_query_reduction_percentages_max-voronoi_query_reduction_percentages_median]), marker='.',
             color='b', ecolor='b',
             elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
plt.errorbar(counts, aabb_query_reduction_percentages_median, np.vstack([aabb_query_reduction_percentages_median-aabb_query_reduction_percentages_min, aabb_query_reduction_percentages_max-aabb_query_reduction_percentages_median]), marker='.', color='r',
             ecolor='r',
             elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
plt.legend(['Triangle Ineq.', 'AABB'])
plt.xlabel('Number of Polytopes')
plt.ylabel('% of Polytopes Evaluated')
plt.title('Dataset Size Scalability with Uniformly Distributed Polytopes')

# plt.title('Nearest Polytope Evaluated Percentage vs. Number of Polytopes')
plt.tight_layout()
plt.savefig('evaluated_percentage.png', dpi=500)

plt.figure(fig_index)
fig_index += 1
plt.subplot(111)
plt.errorbar(counts, counts / 100 * voronoi_query_reduction_percentages_median, counts / 100 * np.vstack(
    [voronoi_query_reduction_percentages_median - voronoi_query_reduction_percentages_min,
     voronoi_query_reduction_percentages_max - voronoi_query_reduction_percentages_median]), marker='.', color='b',
             ecolor='b',
             elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
plt.errorbar(counts, counts / 100 * aabb_query_reduction_percentages_median,
             counts / 100 * np.vstack([aabb_query_reduction_percentages_median - aabb_query_reduction_percentages_min,
                                      aabb_query_reduction_percentages_max - aabb_query_reduction_percentages_median]),
             marker='.', color='r', ecolor='r',
             elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
plt.legend(['Triangle Ineq.', 'AABB'])
plt.xlabel('Number of Polytopes')
plt.ylabel('Number of Polytopes Evaluated')
plt.title('Dataset Size Scalability with Uniformly Distributed Polytopes')

# plt.title('Nearest Polytope Evaluated Percentage vs. Number of Polytopes')
plt.tight_layout()
plt.savefig('evaluated_count.png', dpi=500)


# plt.subplot(212)
# plt.plot(np.log(counts), np.log(voronoi_query_reduction_percentages_median))
# plt.xlabel('$log$ State Dimension')
# plt.ylabel('$log$ % of Zonotopes Evaluated')
# # plt.title('$log$ Voronoi Closest Zonotope Reduction Percentage with %d Zonotopes' %count)
plt.show()