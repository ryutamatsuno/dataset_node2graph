from .base import *


def vis_hid_colored(data, labels, method='UMAP', label_names=None):
    if type(data) is list:
        data = np.array(data)
    elif type(data) is torch.Tensor:
        data = data.detach().numpy()
    if type(labels) is list:
        labels = np.array(labels)
    elif type(labels) is torch.Tensor:
        labels = labels.detach().numpy()

    assert data.shape[0] == labels.shape[0]
    X_reduced = hid22D(data, method)

    x = X_reduced[:, 0]
    y = X_reduced[:, 1]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()

    colors = get_colors(max(labels) + 1)
    makers = get_markers(max(labels) + 1)
    # m_size = 10
    m_size = 20
    idx = np.unique(labels)
    for i in range(len(idx)):
        l = idx[i]
        ix = np.where(labels == l)
        if label_names is not None:
            l = label_names[l]
        ax.scatter(x[ix], y[ix], c=[colors[i]], marker=makers[i], label=l, s=m_size)

    ax.legend()

    fig.tight_layout()
    plt.show()
    return fig


def vis_hid(data, method='PCA'):
    X_reduced = hid22D(data, method)
    vis_scatter(X_reduced[:, 0], X_reduced[:, 1])


def vis_hid_continuous(data, y, method="UMAP"):
    X_reduced = hid22D(data, method)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    cmap = cm.jet

    vmin = min(y)
    vmax = max(y)  # * 1.2

    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, vmin=vmin, vmax=vmax, cmap=cmap, s=10)

    # ax.legend()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # sm._A = []
    # show colorbar
    plt.colorbar(sm)

    fig.tight_layout()
    plt.show()
    return fig


"""
scatter
"""


def ax_scatter(ax, xy, y=None):
    if y is None:
        x = xy[:, 0]
        y = xy[:, 1]
    else:
        x = xy
    ax.scatter(x, y)


def vis_scatter(x, y):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    ax_scatter(ax, x, y)
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    fig.tight_layout()
    plt.show()


def vis_scatter_i(x, y):
    global fig_i
    global ax_i
    if fig_i is None:
        fig_i, ax_i = plt.subplots()
    ax_i.clear()

    ax_scatter(ax_i, x, y)
    fig_i.tight_layout()
    plt.pause(0.01)


def vis_scatter2_i(x1, y1, x2, y2, rs):
    global fig_i
    global ax_i
    if fig_i is None:
        fig_i, ax_i = plt.subplots()

    ax_i.clear()
    ax_i.scatter(x1, y1)
    ax_i.scatter(x2, y2, c='r')

    for i in range(len(x2)):
        # circle = plt.Circle((x2[i], y2[i]), rs[i], color='r', fill=False)
        # ax_i.add_artist(circle)
        e1 = pat.Ellipse((x2[i], y2[i]), width=rs[0, i], height=rs[1, i], fill=False, color="red")
        ax_i.add_patch(e1)

    ax_i.set_xlim(-1, 1)
    ax_i.set_ylim(-1, 1)
    fig_i.tight_layout()
    plt.pause(0.01)
