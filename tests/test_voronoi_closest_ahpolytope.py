from voronoi.voronoi import *
from pypolycontain.utils.random_polytope_generator import *
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ
from visualization.visualize import visualize_2d_voronoi
import matplotlib.pyplot as plt

def test_voronoi_closest_zonotope():
    zonotopes = get_uniform_random_zonotopes(5, dim=2, return_type='zonotope')
    #precompute
    vca = VoronoiClosestPolytope(zonotopes)
    #build query point
    query_point = np.asarray([0,-5])
    np.reshape(query_point,(query_point.shape[0],1))

    #query
    closest_zonotope = vca.find_closest_AHpolytopes(query_point)

    #visualize voronoi
    fig = visualize_2d_voronoi(vca)

    #visualize polytopes
    ax = fig.add_subplot(111)
    fig, ax = visZ(zonotopes, title="", alpha=0.2, fig=fig, ax=ax)
    plt.scatter(query_point[0],query_point[1])

    print('Closest Zonotope: ', closest_zonotope)
    plt.show()

if __name__ == '__main__':
    test_voronoi_closest_zonotope()