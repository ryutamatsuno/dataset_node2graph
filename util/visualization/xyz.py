from .base import *


def vis_xyz(x, y, z):
    """
    :param x: 100
    :param y: 150
    :param z: 100 * 150
    :return:
    """
    # x = np.arange(0, 10, 0.05)
    # y = np.arange(0, 10, 0.05)

    X, Y = np.meshgrid(x, y)
    # Z = np.sin(X) + np.cos(Y)

    fig, ax = plt.subplots()
    # im = ax.pcolormesh(X, Y, z, cmap='viridis')
    im = ax.contour(X, Y, z, cmap='viridis')
    im.set_clim(np.min(z), np.max(z))

    pp = fig.colorbar(im, orientation="vertical")
    pp.set_label('z')

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()
    return fig
