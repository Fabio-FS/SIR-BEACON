import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# functions to prettify plots:

def prettify(ax, 
             xticks = [], yticks = [], xtlabels =[], ytlabels =[], 
             xlabel = '', ylabel = '', title = '', 
             tickfontsize = 8, labelfontsize = 10, titlefontsize = 12, legendfontsize = 8, 
             show_legend_flag = False, xlim = [], ylim = []):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xtlabels, fontsize = tickfontsize)
    ax.set_yticklabels(ytlabels, fontsize = tickfontsize)
    ax.set_xlabel(xlabel, fontsize = labelfontsize)
    ax.set_ylabel(ylabel, fontsize = labelfontsize)
    ax.set_title(title, fontsize = titlefontsize)
    
    if show_legend_flag == True:
        ax.legend(fontsize = legendfontsize)
    if xlim != []:
        ax.set_xlim(xlim)
    if ylim != []:
        ax.set_ylim(ylim)


def discretize_cmaps(name, N):
    c_map = plt.colormaps[name]
    colors = c_map(np.linspace(0, 1, N))
    res = ListedColormap(colors)
    return res

def double_savefig(fig, ax, name, path_Plot_with_labels, path_Plot_without_labels, cbar = False):
    fig.savefig(path_Plot_with_labels + name)
    # remove labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    # remove legend if present
    ax.legend().remove()
    # remove title if present
    ax.set_title('')

    if cbar == False:
        pass
    else:
        cbar.remove()
    fig.savefig(path_Plot_without_labels + name, bbox_inches='tight', pad_inches=0)



colors_pol = ["#0868ac", "#7bccc4", "#a8ddb5"]

colorpalette_1 = ["#b30000", "#e34a33", "#fc8d59", "#fdbb84"]

colorpalette_2 = ["#006837", "#31a354", "#78c679", "#addd8e", "#d9f0a3"]

color_9_greens = ["#f7fcfd", "#e5f5f9", "#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45", "#006d2c", "#00441b"]
color_9_diverg = ["#d73027", "#f46d43", "#fdae61", "#fee090", "#000", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4"]




my_hot = discretize_cmaps('hot', 15)
my_viridis = discretize_cmaps('viridis', 15)



line_styles = ["-.", "--", "-", ":"]

Lx = 18/3.1 # cm
Ly = 17/4 # cm

# convert them in inches

Lx = Lx/2.54
Ly = Ly/2.54