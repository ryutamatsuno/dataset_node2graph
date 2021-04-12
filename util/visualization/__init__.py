import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np

ax_i = None
fig_i = None

from .graph import ax_graph, vis_graph, vis_graph_color, vis_2graphs, vis_2graphs_fix, vis_graph_partition, vis_2graphs_fix_color, vis_graph_value, vis_2graphs_fix_partition
from .heatmap import ax_heatmap, vis_2heatmap, vis_heatmap, vis_heatmap_i
from .xyz import vis_xyz
from .scatter import vis_hid_colored, vis_hid, vis_scatter, vis_hid_continuous, vis_scatter2_i, vis_scatter_i, ax_scatter
from .base import hid22D


def matplot_disable_showing():
    import matplotlib as mpl
    mpl.use('Agg')
    print('matplotlib does not show figures')
    return


# usage
# fig.savefig("img.pdf")


# def vis_2(x, y):
#     fig = plt.figure(figsize=(12, 6))
#     ax = fig.add_subplot(1, 2, 1)
#
#     ax = fig.add_subplot(1, 2, 2)
#
#     plt.tight_layout()
#     plt.show()
#     return fig


def vis_2(f1, f2):
    # usage
    # vis_2(lambda ax: ax_graph(ax, G),lambda ax: ax_cdf(ax, ds))
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    f1(ax)
    ax = fig.add_subplot(1, 2, 2)
    f2(ax)
    plt.tight_layout()
    plt.show()
    return fig


"""
HIgh Dimensional data
"""


#
# def vis_tsne(data):
#     X_reduced = TSNE(n_components=2).fit_transform(data)
#     vis_scatter(X_reduced[:, 0], X_reduced[:, 1])


def vis_plot(x, y):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()

    ax.plot(x, y, 'o-')
    # ax.set_ylim(min(y),min(y)*20)
    fig.tight_layout()
    plt.show()


def ax_cdf(ax, x):
    x = np.array(x)
    x = np.sort(x)
    num_x = len(x)
    ny = [1]
    nx = [min(x)]

    prev_x = None
    for i in range(num_x):
        t = x[i]
        if prev_x == t:
            continue

        nx.append(t)
        ny.append(len(x[x >= t]) / num_x)

        prev_x = t

    e = 1e-10
    nx.append(np.max(x) + e)
    ny.append(0)

    ax.plot(nx, ny, 'o-')


def vis_cdf(x):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()

    ax_cdf(ax, x)

    fig.tight_layout()
    plt.show()
    return fig


"""
Histgrapm
"""


def ax_hist(ax, x):
    # ax.hist(x, bins=50)
    N = x.shape[0]
    n_bins = 40
    hist, bins = np.histogram(x, bins=n_bins, range=(min(x), max(x)))

    # print(hist.shape)
    # print(bins.shape)
    # print(hist)
    # print(bins)

    y = hist
    # x = bins[:-1]
    x = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    x = np.array(x)

    w = (max(x) - min(x)) / 40
    ax.bar(x, y / N, width=w)  #
    # exit()
    # ax.set_xlim(0,2)
    # ax.set_ylim(0,0.10)


def vis_hist_i(x):
    global fig_i
    global ax_i
    if fig_i is None:
        fig_i, ax_i = plt.subplots()

    ax_i.clear()

    ax_hist(ax_i, x)
    # ax_i.set_xlim(-1, 1)
    fig_i.tight_layout()
    plt.pause(0.01)


def vis_hist(x):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()

    ax_hist(ax, x)

    fig.tight_layout()
    plt.show()
    return fig


def vis_confusion(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    cmx_data = confusion_matrix(y_true, y_pred)
    csum = np.sum(cmx_data, axis=1)
    csum[csum == 0] = 1
    cmx_data = cmx_data / np.expand_dims(csum, axis=1)

    vis_heatmap(cmx_data)
