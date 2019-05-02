from voronoi.voronoi import *
from bounding_box.zonotope_tree import ZonotopeTree
from pypolycontain.utils.random_polytope_generator import *
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ
from scipy.spatial import voronoi_plot_2d
from timeit import default_timer
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

def test_uniform_random_zonotope_count(dim=2, counts = np.arange(3, 16, 3)*10, construction_repeats = 1, queries=100,save=True):

    voronoi_precomputation_times = np.zeros([len(counts), construction_repeats])
    voronoi_query_times = np.zeros([len(counts), construction_repeats*queries])
    voronoi_query_reduction_percentages = np.zeros([len(counts), construction_repeats*queries])

    aabb_precomputation_times = np.zeros([len(counts), construction_repeats])
    aabb_query_times = np.zeros([len(counts), construction_repeats*queries])
    aabb_query_reduction_percentages = np.zeros([len(counts), construction_repeats*queries])


    for cr_index in range(construction_repeats):
        print('Repetition %d' %cr_index)
        for count_index, count in enumerate(counts):
            print('Testing %d zonotopes...' % count)
            zonotopes = get_uniform_random_zonotopes(count, dim=dim, generator_range=count * 1.2,return_type='zonotope')

            #test voronoi
            construction_start_time = default_timer()
            vcp = VoronoiClosestPolytope(zonotopes)
            voronoi_precomputation_times[count_index, cr_index] = default_timer()-construction_start_time
            #query
            for query_index in range(queries):
                query_point = (np.random.rand(dim) - 0.5) * count * 5 #random query point
                query_start_time = default_timer()
                best_zonotope, best_distance, evaluated_zonotopes = vcp.find_closest_polytope(query_point, return_intermediate_info=True)
                voronoi_query_times[count_index,cr_index*queries+query_index] = default_timer()-query_start_time
                voronoi_query_reduction_percentages[count_index, cr_index*queries+query_index] = len(evaluated_zonotopes)*100./count

            #test aabb
            construction_start_time = default_timer()
            zono_tree = ZonotopeTree(zonotopes)
            aabb_precomputation_times[count_index, cr_index] = default_timer()-construction_start_time
            #query
            for query_index in range(queries):
                query_point = (np.random.rand(dim) - 0.5) * count * 5 #random query point
                query_start_time = default_timer()
                best_zonotope, best_distance, evaluated_zonotopes = zono_tree.find_closest_zonotopes(query_point, return_intermediate_info=True)
                aabb_query_times[count_index,cr_index*queries+query_index] = default_timer()-query_start_time
                aabb_query_reduction_percentages[count_index, cr_index*queries+query_index] = len(evaluated_zonotopes)*100./count

    voronoi_precomputation_times_avg = np.mean(voronoi_precomputation_times, axis=1)
    voronoi_precomputation_times_std = np.std(voronoi_precomputation_times, axis=1)

    voronoi_query_times_avg = np.mean(voronoi_query_times, axis=1)
    voronoi_query_times_std = np.std(voronoi_query_times, axis=1)

    voronoi_query_reduction_percentages_avg =np.mean(voronoi_query_reduction_percentages, axis=1)
    voronoi_query_reduction_percentages_std = np.std(voronoi_query_reduction_percentages, axis=1)


    aabb_precomputation_times_avg = np.mean(aabb_precomputation_times, axis=1)
    aabb_precomputation_times_std = np.std(aabb_precomputation_times, axis=1)

    aabb_query_times_avg = np.mean(aabb_query_times, axis=1)
    aabb_query_times_std = np.std(aabb_query_times, axis=1)

    aabb_query_reduction_percentages_avg =np.mean(aabb_query_reduction_percentages, axis=1)
    aabb_query_reduction_percentages_std = np.std(aabb_query_reduction_percentages, axis=1)


    #plots
    fig_index = 0
    plt.figure(fig_index)
    fig_index+=1
    plt.subplot(111)
    plt.errorbar(counts,voronoi_precomputation_times_avg,voronoi_precomputation_times_std,marker='.',color='b', ecolor='b',elinewidth=0.3,capsize=2,linewidth=0.5,markersize=7)
    plt.errorbar(counts,aabb_precomputation_times_avg,aabb_precomputation_times_std,marker='.',color='r',ecolor='r',elinewidth=0.3,capsize=2,linewidth=0.5,markersize=7)
    plt.legend(['Centroid Dist.', 'AABB'])

    plt.xlabel('Zonotope Count')
    plt.ylabel('Precomputation Time (s)')
    plt.title('Closest Zonotope Precomputation Time in %d-D' %dim)
    #
    # plt.subplot(212)
    # plt.plot(np.log(counts),np.log(voronoi_precomputation_times_avg))
    # plt.xlabel('$log$ Zonotope Count')
    # plt.ylabel('$log$ Precomputation Time (s)')
    # plt.title('$log$ Voronoi Closest Zonotope Precomputation Time in %d-D' %dim)
    # plt.tight_layout()
    if save:
        plt.savefig('precomputation_time' + str(default_timer()) + '.png', dpi=500)

    plt.figure(fig_index)
    fig_index+=1
    plt.subplot(111)
    plt.errorbar(counts,voronoi_query_times_avg,voronoi_query_times_std,marker='.',color='b',ecolor='b',elinewidth=0.3,capsize=2,linewidth=0.5,markersize=7)
    plt.errorbar(counts,aabb_query_times_avg,aabb_query_times_std,marker='.',color='r', ecolor='r',elinewidth=0.3,capsize=2,linewidth=0.5,markersize=7)
    plt.legend(['Centroid Dist.', 'AABB'])
    plt.xlabel('Zonotope Count')
    plt.ylabel('Query Time (s)')
    plt.title('Closest Zonotope Single Query Time in %d-D' %dim)
    #
    # plt.subplot(212)
    # plt.plot(np.log(counts),np.log(aabb_query_times_avg))
    # plt.xlabel('$log$ Zonotope Count')
    # plt.ylabel('$log$ Query Time (s)')
    # plt.title('$log$ Voronoi Closest Zonotope Single Query Time in %d-D' %dim)
    # plt.tight_layout()
    if save:
        plt.savefig('query_time' + str(default_timer()) + '.png', dpi=500)

    plt.figure(fig_index)
    fig_index+=1
    plt.subplot(111)
    plt.errorbar(counts,voronoi_query_reduction_percentages_avg,voronoi_query_reduction_percentages_std,marker='.',color='b', ecolor='b',elinewidth=0.3,capsize=2,linewidth=0.5,markersize=7)
    plt.errorbar(counts,aabb_query_reduction_percentages_avg,aabb_query_reduction_percentages_std,marker='.',color='r',ecolor='r',elinewidth=0.3,capsize=2,linewidth=0.5,markersize=7)
    plt.legend(['Centroid Dist.', 'AABB'])

    plt.xlabel('Zonotope Count')
    plt.ylabel('% of Zonotopes Evaluated')
    plt.title('Closest Zonotope Reduction Percentage in %d-D' %dim)
    # plt.ylim(ymin=0)
    #
    # plt.subplot(212)
    # plt.plot(np.log(counts),np.log(voronoi_query_reduction_percentages_avg))
    # plt.xlabel('$log$ Zonotope Count')
    # plt.ylabel('$log$ % of Zonotopes Evaluated')
    # plt.title('$log$ Voronoi Closest Zonotope Reduction Percentage in %d-D' %dim)
    # plt.tight_layout()
    if save:
        plt.savefig('reduction_percentage' + str(default_timer()) + '.png', dpi=500)

    else:
        plt.show()



def test_uniform_random_zonotope_dim(count=100, dims=np.arange(2, 11, 1), construction_repeats=1, queries=100, save=True):
    voronoi_precomputation_times = np.zeros([len(dims), construction_repeats])
    voronoi_query_times = np.zeros([len(dims), construction_repeats * queries])
    voronoi_query_reduction_percentages = np.zeros([len(dims), construction_repeats * queries])
    aabb_precomputation_times = np.zeros([len(dims), construction_repeats])
    aabb_query_times = np.zeros([len(dims), construction_repeats * queries])
    aabb_query_reduction_percentages = np.zeros([len(dims), construction_repeats * queries])

    for cr_index in range(construction_repeats):
        print('Repetition %d' % cr_index)
        for dim_index, dim in enumerate(dims):
            print('Testing zonotopes in %d-D...' % dim)
            # generate random zonotopes
            zonotopes = get_uniform_random_zonotopes(count, dim=dim, generator_range=count * 1.2,
                                                     centroid_range=count * 10, return_type='zonotope')
            #test voronoi
            construction_start_time = default_timer()
            vcp = VoronoiClosestPolytope(zonotopes)
            voronoi_precomputation_times[dim_index, cr_index] = default_timer() - construction_start_time
            # query
            for query_index in range(queries):
                query_point = (np.random.rand(dim) - 0.5) * count * 5  # random query point
                query_start_time = default_timer()
                best_zonotope, best_distance, evaluated_zonotopes = vcp.find_closest_polytope(query_point,
                                                                                              return_intermediate_info=True)
                voronoi_query_times[dim_index, cr_index * queries + query_index] = default_timer() - query_start_time
                voronoi_query_reduction_percentages[dim_index, cr_index * queries + query_index] = len(
                    evaluated_zonotopes) * 100. / count

            #test aabb
            construction_start_time = default_timer()
            zono_tree = ZonotopeTree(zonotopes)
            aabb_precomputation_times[dim_index, cr_index] = default_timer() - construction_start_time
            # query
            for query_index in range(queries):
                query_point = (np.random.rand(dim) - 0.5) * count * 5  # random query point
                query_start_time = default_timer()
                best_zonotope, best_distance, evaluated_zonotopes = zono_tree.find_closest_zonotopes(query_point, return_intermediate_info=True)
                aabb_query_times[dim_index, cr_index * queries + query_index] = default_timer() - query_start_time
                aabb_query_reduction_percentages[dim_index, cr_index * queries + query_index] = len(evaluated_zonotopes) * 100. / count

    voronoi_precomputation_times_avg = np.mean(voronoi_precomputation_times, axis=1)
    voronoi_precomputation_times_std = np.std(voronoi_precomputation_times, axis=1)

    voronoi_query_times_avg = np.mean(voronoi_query_times, axis=1)
    voronoi_query_times_std = np.std(voronoi_query_times, axis=1)

    voronoi_query_reduction_percentages_avg = np.mean(voronoi_query_reduction_percentages, axis=1)
    voronoi_query_reduction_percentages_std = np.std(voronoi_query_reduction_percentages, axis=1)

    aabb_precomputation_times_avg = np.mean(aabb_precomputation_times, axis=1)
    aabb_precomputation_times_std = np.std(aabb_precomputation_times, axis=1)

    aabb_query_times_avg = np.mean(aabb_query_times, axis=1)
    aabb_query_times_std = np.std(aabb_query_times, axis=1)

    aabb_query_reduction_percentages_avg = np.mean(aabb_query_reduction_percentages, axis=1)
    aabb_query_reduction_percentages_std = np.std(aabb_query_reduction_percentages, axis=1)



    # plots
    fig_index = 0
    plt.figure(fig_index)
    fig_index += 1
    plt.subplot(111)
    plt.errorbar(dims, voronoi_precomputation_times_avg, voronoi_precomputation_times_std, marker='.', color = 'b', ecolor='b', elinewidth=0.3,
                 capsize=2, linewidth=0.5, markersize=7)
    plt.errorbar(dims, aabb_precomputation_times_avg, aabb_precomputation_times_std, marker='.', color='r', ecolor='r', elinewidth=0.3,
                 capsize=2, linewidth=0.5, markersize=7)
    plt.legend(['Centroid Dist.', 'AABB'])
    plt.xlabel('State Dimension')
    plt.ylabel('Precomputation Time (s)')
    plt.title('Closest Zonotope Precomputation Time with %d Zonotopes' % count)
    #
    # plt.subplot(212)
    # plt.plot(dims, np.log(voronoi_precomputation_times_avg))
    # plt.xlabel('$log$ State Dimension')
    # plt.ylabel('$log$ Precomputation Time (s)')
    # # plt.title('Voronoi Closest Zonotope Precomputation Time with %d Zonotopes' %count)
    # plt.tight_layout()

    if save:
        plt.savefig('precomputation_time' + str(default_timer()) + '.png', dpi=500)

    plt.figure(fig_index)
    fig_index += 1
    plt.subplot(111)
    plt.errorbar(dims, voronoi_query_times_avg, voronoi_query_times_std, marker='.', color='b', ecolor='b', elinewidth=0.3, capsize=2,
                 linewidth=0.5, markersize=7)
    plt.errorbar(dims, aabb_query_times_avg, aabb_query_times_std, marker='.', color='r', ecolor='r', elinewidth=0.3, capsize=2,
                 linewidth=0.5, markersize=7)
    plt.legend(['Centroid Dist.', 'AABB'])
    plt.xlabel('State Dimension')
    plt.ylabel('Query Time (s)')
    plt.title('Closest Zonotope Single Query Time with %d Zonotopes' % count)

    # plt.subplot(212)
    # plt.plot(np.log(dims), np.log(voronoi_query_times_avg))
    # plt.xlabel('$log$ State Dimension')
    # plt.ylabel('$log$ Query Time (s)')
    # # plt.title('$log$ Voronoi Closest Zonotope Single Query Time with %d Zonotopes' %count)
    # plt.tight_layout()

    if save:
        plt.savefig('query_time' + str(default_timer()) + '.png', dpi=500)

    plt.figure(fig_index)
    fig_index += 1
    plt.subplot(111)
    plt.errorbar(dims, voronoi_query_reduction_percentages_avg, voronoi_query_reduction_percentages_std, marker='.', color='b',ecolor='b',
                 elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
    plt.errorbar(dims, aabb_query_reduction_percentages_avg, aabb_query_reduction_percentages_std, marker='.', color='r', ecolor='r',
                 elinewidth=0.3, capsize=2, linewidth=0.5, markersize=7)
    plt.legend(['Centroid Dist.', 'AABB'])
    plt.xlabel('State Dimension')
    plt.ylabel('% of Zonotopes Evaluated')
    plt.title('Closest Zonotope Reduction Percentage with %d Zonotopes' % count)

    # plt.subplot(212)
    # plt.plot(np.log(dims), np.log(voronoi_query_reduction_percentages_avg))
    # plt.xlabel('$log$ State Dimension')
    # plt.ylabel('$log$ % of Zonotopes Evaluated')
    # # plt.title('$log$ Voronoi Closest Zonotope Reduction Percentage with %d Zonotopes' %count)
    # plt.tight_layout()

    if save:
        plt.savefig('reduction_percentage' + str(default_timer()) + '.png', dpi=500)
    else:
        plt.show()


if __name__ == '__main__':
    # print('time_against_count(dim=6, counts=np.arange(2, 11, 2) * 100, construction_repeats=3, queries=100)')
    # test_uniform_random_zonotope_count(dim=6, counts=np.arange(2, 11, 2) * 100, construction_repeats=3, queries=100)
    print('test_uniform_random_zonotope_dim(count=500, dims=np.arange(2, 11, 1), construction_repeats=3, queries=100)')
    test_uniform_random_zonotope_dim(count=500, dims=np.arange(2, 11, 1), construction_repeats=3, queries=100)
    # test_voronoi_closest_zonotope(100, save=False)