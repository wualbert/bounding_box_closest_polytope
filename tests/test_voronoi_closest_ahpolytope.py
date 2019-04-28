from voronoi.voronoi import *
from pypolycontain.utils.random_polytope_generator import *
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ
from scipy.spatial import voronoi_plot_2d
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"


def test_voronoi_closest_zonotope():
    zonotope_count = 30
    zonotopes = get_uniform_random_zonotopes(30, dim=2, generator_range=zonotope_count*1.2,return_type='zonotope')
    #precompute
    vca = VoronoiClosestPolytope(zonotopes)
    #build query point
    query_point = (np.random.rand(2)-0.5)*zonotope_count*3
    np.reshape(query_point,(query_point.shape[0],1))

    #query
    evaluated_zonotopes = vca.find_closest_AHpolytopes(query_point, k_closest=np.inf)
    print('Checked %d zonotopes' %len(evaluated_zonotopes))
    closest_zonotope = evaluated_zonotopes[0]

    #visualize voronoi
    fig = voronoi_plot_2d(vca.centroid_voronoi, point_size=2,show_vertices=False, line_alpha=0.4, line_width=1)

    #visualize polytopes
    ax = fig.add_subplot(111)
    fig, ax = visZ([closest_zonotope], title="", fig=fig, ax=ax, alpha=0.8)
    fig, ax = visZ(zonotopes, title="", alpha=0.1, fig=fig, ax=ax)
    plt.scatter(query_point[0],query_point[1], facecolor='red', s=4)

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Closest Zonotope with Voronoi Diagram')
    print('Closest Zonotope: ', closest_zonotope)
    plt.show()

if __name__ == '__main__':
    test_voronoi_closest_zonotope()