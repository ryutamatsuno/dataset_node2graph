import argparse

import networkx as nx
import torch
from tqdm import tqdm

from dataset.GraphData import GraphData
from node_dataset.NodeData import NodeData
import time
from distutils.util import strtobool

def naive_khop_neighbors(x: int, G: nx.Graph, k: int):
    if k == 1:
        return set(nx.neighbors(G, x)).union({x})

    neigbors = set(nx.neighbors(G, x))
    neigbors.add(x)
    for i in range(k - 1):
        for y in neigbors:
            ynei = set(nx.neighbors(G, y))
            neigbors = neigbors.union(ynei)
    return neigbors


def khop_neighbors(x: int, G: nx.Graph, k: int):
    if k == 1:
        return set(nx.neighbors(G, x)).union({x})

    neigbors = set(nx.neighbors(G, x))
    neigbors.add(x)
    edge_neigbors = nx.neighbors(G, x)
    for i in range(k - 1):
        next_edge_neibors = set()
        for y in edge_neigbors:
            ynei = set(nx.neighbors(G, y))
            next_edge_neibors = next_edge_neibors.union(ynei - neigbors)
            neigbors = neigbors.union(ynei)
        edge_neigbors = next_edge_neibors

    n2 = naive_khop_neighbors(x, G, k)
    assert neigbors == n2, (len(neigbors), len(n2))
    return neigbors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='wisconsin', help='')
    parser.add_argument('--k', type=int, default=1, help='')
    parser.add_argument('--feature', action='store_true')
    args = parser.parse_args()
    print(args)

    print('Node data:', args.dataset)
    nodedata = NodeData.load(args.dataset)

    G = nx.Graph()
    G.add_edges_from(nodedata.raw_edge_list.transpose(0, 1).long().tolist())

    assert nx.is_connected(G), 'Graph is not connected'

    Es = []
    labels = []
    features = []

    nodes = sorted(nx.nodes(G))
    N = len(nodes)

    for i in tqdm(range(N)):
        x = nodes[i]

        neighbors = khop_neighbors(x, G, args.k)
        if args.k == 1:
            assert len(neighbors) == nx.degree(G, x) + 1

        H = nx.induced_subgraph(G, neighbors)
        label = nodedata.labels[i]
        Hnodes = sorted(nx.nodes(H))
        # print(Hnodes)

        remap = {}
        for q in range(len(Hnodes)):
            remap[Hnodes[q]] = q

        H = nx.relabel_nodes(H, remap)
        E = list(nx.edges(H))
        feature = nodedata.features[Hnodes]

        # vis_graph(H)
        E = torch.tensor(E, dtype=torch.long).transpose(0, 1)
        # feature = torch.tensor(feature, dtype=torch.float)
        Es.append(E)
        labels.append(label)
        features.append(feature)

    print('convert done')

    labels = torch.tensor(labels, dtype=torch.long)
    graphdata = GraphData(Es, labels, features)

    dataset_name = args.dataset
    if args.k > 1:
        dataset_name = dataset_name + str(args.k)
    graphdata.save(dataset_name, args.feature)

    exit()
