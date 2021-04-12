from .base import *
import mpl_toolkits.axes_grid1

ax_i = None
fig_i = None


def ax_heatmap(ax, data, show_value=True):
    W, H = data.shape  # len(x), len(y)

    # # 0,1
    # im = ax.imshow(data, interpolation='nearest', cmap=cm.copper)
    im = ax.imshow(data, interpolation='nearest', cmap=cm.GnBu)
    # im = ax.imshow(data, interpolation='nearest', cmap=cm.jet)

    im.set_clim(0, 1)

    # -1,1
    # im = ax.imshow(data, interpolation='nearest', cmap=cm.bwr)
    # im.set_clim(-1, 1)

    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    # plt.colorbar(im, cax=cax)

    # bar = fig.colorbar(im, ticks=[np.min(data), 0, np.max(data)])

    cbar = plt.colorbar(im, ticks=[0, 0.2,0.4,0.6,0.8, 1], cax=cax)

    # cm.copper
    # cm.jet
    # cm.winter
    # cm.coolwarm
    # cm.inferno
    # https://matplotlib.org/examples/color/colormaps_reference.html
    # cm.hot

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # show value
    if max(W, H) < 20 and show_value:
        for i in range(H):
            for j in range(W):
                ax.text(j, i, "%.2f" % data[i, j], ha="center", va="center", color="w")


def vis_heatmap(data, show_value=True):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    ax_heatmap(ax, data, show_value=show_value)
    fig.tight_layout()
    plt.show()
    return fig


def vis_2heatmap(data1, data2):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax_heatmap(ax, data1)
    ax = fig.add_subplot(1, 2, 2)
    ax_heatmap(ax, data2)

    fig.tight_layout()
    plt.show()
    return fig


def vis_heatmap_i(data):
    global fig_i
    global ax_i
    if fig_i is None:
        fig_i, ax_i = plt.subplots()

    ax_i.clear()
    ax_heatmap(ax_i, data)

    fig_i.tight_layout()
    plt.pause(0.001)
