from .base import *


def get_partition_index(partition, i):
    for k in range(len(partition)):
        if i in partition[k]:
            return k
    raise ValueError()


def ax_graph(ax, A, pos=None, with_labels=True):
    """
    visualize graph and show window
    A is nx.Graph or numpy matrix
    """

    if isinstance(A, type(nx.Graph())):
        G = A
        N = len(G)
    else:
        G = nx.from_numpy_matrix(A)
        N = A.shape[0]

    options = {
        'edge_color': 'black',
        'width': 1,
        'with_labels': with_labels,  # [str(i) for i in range(N)],
        'font_weight': 'regular',
        'node_size': 800,
        'font_size': 20,
        'node_color': 'orange',
    }
    if pos is None:
        pos = nx.spring_layout(G, k=1, iterations=200)
    nx.draw(G, pos=pos, ax=ax, **options)


def vis_graph_partition(G: nx.Graph, pt: [set()], pos=None, label='both'):
    colors = {}
    for i in sorted(nx.nodes(G)):
        colors[i] = get_partition_index(pt, i)
    return vis_graph_color(G, colors, pos=pos, label=label)


def vis_graph_color(A, node2color, pos=None, label='both'):
    """
    visualize graph and show window
    A is nx.Graph or numpy matrix
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    if isinstance(A, type(nx.Graph())):
        G = A
        N = len(G)
    else:
        G = nx.from_numpy_matrix(A)
        N = A.shape[0]

    if type(node2color) is list:
        # assume sorted(nx.nodes) order
        temp = {}
        for i, n in enumerate(sorted(nx.nodes(G))):
            temp[n] = node2color[i]
        node2color = temp

    n_col = max(node2color.values()) - min(node2color.values()) + 1

    colors = get_colors(max(node2color.values()) + 1)

    # color_map = []
    # for i in range(N):
    #     color_map.append(colors[color_indices[i]])

    color_list = []  # <- node order should be nx.nodes
    for n in nx.nodes(G):
        color_list.append(colors[node2color[n]])

    options = {
        'edge_color': 'black',
        'width': 1,
        'font_weight': 'regular',
        'font_size': 15,
        'node_color': color_list,
        'node_size': 800,
    }

    if label == 'node':
        options['with_labels'] = True
        options['node_size'] = 800
    elif label == 'color':
        # values
        n2l = {}
        for n in nx.nodes(G):
            n2l[n] = node2color[n]
        options['with_labels'] = True
        options['labels'] = n2l
        options['node_size'] = 800
    elif label == 'both':
        # values
        n2l = {}
        for n in nx.nodes(G):
            n2l[n] = str(n) + ":" + str(node2color[n])
        options['with_labels'] = True
        options['labels'] = n2l
        options['node_size'] = 800
        options['font_size'] = 10
    else:
        options['with_labels'] = False
        # options['node_size'] = 150

    if pos is None:
        pos = nx.spring_layout(G, k=0.5, iterations=200)
    nx.draw(G, pos, ax=ax, **options)
    plt.show()
    return fig


def vis_graph_value(A, node2color, label='both'):
    """
    visualize graph and show window
    A is nx.Graph or numpy matrix
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    if isinstance(A, type(nx.Graph())):
        G = A
        N = len(G)
    else:
        G = nx.from_numpy_matrix(A)
        N = A.shape[0]

    if type(node2color) is list:
        # assume sorted(nx.nodes) order
        temp = {}
        for i, n in enumerate(sorted(nx.nodes(G))):
            temp[n] = node2color[i]
        node2color = temp

    color_list = []  # <- node order should be nx.nodes
    for n in nx.nodes(G):
        color_list.append(node2color[n])

    cmap = cm.jet  # plt.cm.coolwarm
    vmin = min(color_list)
    vmax = max(color_list)
    options = {
        'edge_color': 'black',
        'width': 1,
        'font_weight': 'regular',
        'font_size': 15,
        'node_color': color_list,
        'cmap': cmap,
        'vmin': vmin,
        'vmax': vmax
    }

    if label == 'node':
        options['with_labels'] = True
        options['node_size'] = 800
    elif label == 'color':
        # values
        n2l = {}
        for n in nx.nodes(G):
            n2l[n] = node2color[n]
        options['with_labels'] = True
        options['labels'] = n2l
        options['node_size'] = 800
    elif label == 'both':
        # values
        n2l = {}
        for n in nx.nodes(G):
            n2l[n] = str(n) + ":" + str(node2color[n])
        options['with_labels'] = True
        options['labels'] = n2l
        options['node_size'] = 800
        options['font_size'] = 10
    else:
        options['with_labels'] = False
        options['node_size'] = 150

    nx.draw(G, pos=nx.spring_layout(G, k=0.5, iterations=200), ax=ax, **options)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(sm)

    plt.show()
    return fig


def compute_position(G):
    assert type(G) is nx.Graph, ('the input should be nx.Graph:', type(G))
    return nx.spring_layout(G, k=1, iterations=200)


def vis_graph(A, pos=None, with_labels=True):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    ax_graph(ax, A, pos, with_labels=with_labels)
    plt.show()
    return fig


def vis_2graphs(A1, A2):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax_graph(ax, A1)
    ax = fig.add_subplot(1, 2, 2)
    ax_graph(ax, A2)
    plt.tight_layout()
    plt.show()
    return fig


def vis_2graphs_fix(Apred, Atrue):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)

    if isinstance(Apred, type(nx.Graph())):
        G = Apred
    else:
        G = nx.from_numpy_matrix(Apred)

    options = {
        'edge_color': 'black',
        'width': 1,
        'with_labels': True,  # [str(i) for i in range(N)],
        'font_weight': 'regular',
        'node_size': 800,
        'font_size': 20,
        'node_color': 'orange',
    }
    pos = nx.spring_layout(G, k=1, iterations=200)
    nx.draw(G, pos=pos, ax=ax, **options)
    ax = fig.add_subplot(1, 2, 2)

    if isinstance(Atrue, type(nx.Graph())):
        G = Atrue
    else:
        G = nx.from_numpy_matrix(Atrue)
    nx.draw(G, pos=pos, ax=ax, **options)

    plt.tight_layout()
    plt.show()
    return fig


def vis_2graphs_fix_partition(G1: nx.Graph, pt1: [set()], G2, pt2):
    colors1 = {}
    for i in sorted(nx.nodes(G1)):
        colors1[i] = get_partition_index(pt1, i)
    colors2 = {}
    for i in sorted(nx.nodes(G2)):
        colors2[i] = get_partition_index(pt2, i)
    vis_2graphs_fix_color(G1, colors1, G2, colors2)
    # vis_graph_color(G, colors, label='both')


def vis_2graphs_fix_color(A1, node2color1, A2, node2color2, label='color'):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)

    if isinstance(A1, type(nx.Graph())):
        G = A1
    else:
        G = nx.from_numpy_matrix(A1)

    if type(node2color1) is list:
        # assume sorted(nx.nodes) order
        temp = {}
        for i, n in enumerate(sorted(nx.nodes(G))):
            temp[n] = node2color1[i]
        node2color1 = temp

    # colors = ['red', 'blue', 'green', 'grey', 'aqua', 'orange', 'purple', 'sage']
    colors = get_colors(max(node2color1.values()) + 1)
    color_list = []  # <- node order should be nx.nodes
    for n in nx.nodes(G):
        color_list.append(colors[node2color1[n]])

    options = {
        'edge_color': 'black',
        'width': 1,
        'font_weight': 'regular',
        'font_size': 15,
        'node_color': color_list,
    }

    if label == 'node':
        options['with_labels'] = True
        options['node_size'] = 800
    elif label == 'color':
        # values
        n2l = {}
        for n in nx.nodes(G):
            n2l[n] = node2color1[n]
        options['with_labels'] = True
        options['labels'] = n2l
        options['node_size'] = 800
    else:
        options['with_labels'] = False
        options['node_size'] = 150

    pos = nx.spring_layout(G, k=1, iterations=200)
    nx.draw(G, pos=pos, ax=ax, **options)
    ax = fig.add_subplot(1, 2, 2)

    # if isinstance(A2, type(nx.Graph())):
    #     G = A2
    # else:
    #     G = nx.from_numpy_matrix(A2)

    if type(node2color2) is list:
        # assume sorted(nx.nodes) order
        temp = {}
        for i, n in enumerate(sorted(nx.nodes(G))):
            temp[n] = node2color2[i]
        node2color2 = temp

    colors = get_colors(max(node2color2.values()) + 1)
    color_list = []  # <- node order should be nx.nodes
    for n in nx.nodes(G):
        color_list.append(colors[node2color2[n]])
    options['node_color'] = color_list

    if label == 'color':
        # values
        n2l = {}
        for n in nx.nodes(G):
            n2l[n] = node2color2[n]
        # options['with_labels'] = True
        options['labels'] = n2l

    nx.draw(G, pos=pos, ax=ax, **options)

    plt.tight_layout()
    plt.show()
    return fig
