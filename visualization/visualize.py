import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from scipy.spatial import voronoi_plot_2d
from pypolycontain.visualization.visualize_2D import visualize_2D_zonotopes as visZ


def visualize_boxes(box_list, dim_x = 0, dim_y = 1, xlim=None, ylim=None, ax = None, fig = None,alpha=None,fill=False,linewidth=3):
    if ax is None:
        fig, ax = plt.subplots(1)

    for box in box_list:
        x = box.u[dim_x]
        y = box.u[dim_y]
        width = box.v[dim_x] - box.u[dim_x]
        height = box.v[dim_y] - box.u[dim_y]
        ec = np.random.rand(3,)
        if fill:
            rect = patches.Rectangle((x,y), width, height,linewidth=linewidth,edgecolor=box.color,facecolor=box.color,alpha=alpha)
        else:
            rect = patches.Rectangle((x, y), width, height, linewidth=linewidth, edgecolor=box.color, facecolor='none',alpha=alpha)
        ax.add_patch(rect)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    return fig,ax

def visualize_box_nodes(box_nodes_list, dim_x = 0, dim_y = 1, xlim=None, ylim=None, ax = None, fig = None,alpha=None,fill=False,linewidth=3):
    box_list = []
    for box_node in box_nodes_list:
        box_list.append(box_node.box)
    return visualize_boxes(box_list,dim_x = dim_x, dim_y = dim_y,linewidth=linewidth, xlim=xlim, ylim=ylim, ax = ax, fig = fig,alpha=alpha, fill=fill)

