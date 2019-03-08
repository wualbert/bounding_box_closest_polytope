import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection


def visualize_boxes(box_list, dim_x = 0, dim_y = 1,xlim=(0,10),ylim=(0,10)):
    fig,ax1 = plt.subplots(1)
    ax1.set_xlim(xlim[0],xlim[1])
    ax1.set_ylim(ylim[0],ylim[1])
    for box in box_list:
        x = box.u[dim_x]
        y = box.u[dim_y]
        width = box.v[dim_x] - box.u[dim_x]
        height = box.v[dim_y] - box.u[dim_y]
        print(x,y,width,height)
        ec = np.random.rand(3,)
        rect = patches.Rectangle((x,y), width, height,linewidth=2,edgecolor=ec,facecolor='none')
        ax1.add_patch(rect)

    plt.show()
    return plt