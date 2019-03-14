import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection


def visualize_boxes(box_list, dim_x = 0, dim_y = 1, xlim=(0,10), ylim=(0,10), ax = None, fig = None,alpha=None,fill=False):
    if ax is None:
        fig, ax = plt.subplots(1)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
    for box in box_list:
        x = box.u[dim_x]
        y = box.u[dim_y]
        width = box.v[dim_x] - box.u[dim_x]
        height = box.v[dim_y] - box.u[dim_y]
        ec = np.random.rand(3,)
        if fill:
            rect = patches.Rectangle((x,y), width, height,linewidth=3,edgecolor=box.color,facecolor=box.color,alpha=alpha)
        else:
            rect = patches.Rectangle((x, y), width, height, linewidth=3, edgecolor=box.color, facecolor='none',alpha=alpha)
        ax.add_patch(rect)

    return fig,ax

def visualize_box_nodes(box_nodes_list, dim_x = 0, dim_y = 1, xlim=(0,10), ylim=(0,10), ax = None, fig = None,alpha=None,fill=False):
    box_list = []
    for box_node in box_nodes_list:
        box_list.append(box_node.box)
    return visualize_boxes(box_list,dim_x = dim_x, dim_y = dim_y, xlim=xlim, ylim=ylim, ax = ax, fig = fig,alpha=alpha, fill=fill)

