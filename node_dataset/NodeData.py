import os
import networkx as nx
import numpy as np
import torch
import math
import random

# util.SparseTensor import SparseTensor

TOPDIR = 'node_dataset/'


def add_self_loops(edge_list, size) -> torch.Tensor:
    i = torch.arange(size, dtype=torch.int64).view(1, -1)
    self_loops = torch.cat((i, i), dim=0)
    edge_list = torch.cat((edge_list, self_loops), dim=1)
    return edge_list


def get_degree(edge_list) -> torch.Tensor:
    row, col = edge_list
    deg = torch.bincount(row)
    return deg


def normalize_adj(edge_list):
    deg = get_degree(edge_list)
    row, col = edge_list
    deg_inv_sqrt = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    weight = torch.ones(edge_list.size(1))
    v = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj


def edgelist2adj(edge_list):
    v = torch.ones(edge_list.size(1))
    adj = torch.sparse.FloatTensor(edge_list, v)
    return adj


def index_to_mask(index, size) -> torch.Tensor:
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


def split_by_ratio(labels: torch.Tensor, train_ratio, valid_ratio, seed: int) -> (
        torch.Tensor, torch.Tensor, torch.Tensor):
    """
    uniformly random
    :param labels:
    :param train_ratio:
    :param valid_ratio:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    N = labels.shape[0]

    idx = np.random.permutation(np.arange(N, dtype=np.long))
    idx = torch.tensor(idx, dtype=torch.long)
    num_train = int(N * train_ratio)
    num_valid = int(N * valid_ratio)
    train_mask = index_to_mask(idx[:num_train], labels.size(0))
    idx = idx[num_train:]
    valid_mask = index_to_mask(idx[:num_valid], labels.size(0))
    idx = idx[num_valid:]
    tests_mask = index_to_mask(idx, labels.size(0))

    return train_mask, valid_mask, tests_mask


def preprocess_features(features: torch.Tensor):
    rowsum = features.sum(dim=1, keepdim=True)
    rowsum[rowsum == 0] = 1
    features = features / rowsum
    return features


def label_counts(data, mask=None) -> np.ndarray:
    counts = np.zeros(data.num_classes, dtype=np.int)
    labels = data.labels.numpy()
    for i in range(labels.shape[0]):
        if mask is not None:
            if not mask[i]:
                continue
        counts[int(labels[i])] += 1
    return counts


def label_percentage(data, mask=None) -> np.ndarray:
    counts = label_counts(data, mask).astype(np.float)
    counts = counts * 100 / np.sum(counts)
    return counts


def homophily(G: nx.Graph, labels: torch.Tensor) -> float:
    N = labels.shape[0]
    labels = labels.numpy()

    nodes = nx.nodes(G)
    homo = 0
    for v in nodes:
        neighbors = G.neighbors(v)

        d = nx.degree(G, v)
        lv = labels[v]
        homo_count = 0
        for n in neighbors:
            ln = labels[n]
            if lv == ln:
                homo_count += 1
        homo += homo_count / d

    return homo / N


class NodeData(object):

    @staticmethod
    def load(dataname):

        top_dir = TOPDIR + dataname

        labels = np.loadtxt(top_dir + '/labels.csv', dtype=np.int, delimiter=",")
        features = np.loadtxt(top_dir + '/features.csv', dtype=np.float, delimiter=",")
        edge_list = np.loadtxt(top_dir + '/edge_list.edg', dtype=np.int, delimiter=",").transpose()

        # convert to tensor
        labels = torch.tensor(labels, dtype=torch.long)
        features = torch.tensor(features, dtype=torch.float)
        if features.dim() == 1: features = features.unsqueeze(1)
        edge_list = torch.tensor(edge_list, dtype=torch.long)

        data = NodeData(edge_list, features, labels)
        data.dataname = dataname
        return data

    def save(self, dataname):

        top_dir = TOPDIR + dataname
        if not os.path.exists(top_dir):
            os.mkdir(top_dir)
        np.savetxt(top_dir + '/labels.csv', self.labels.numpy(), '%d', delimiter=",")
        np.savetxt(top_dir + '/features.csv', self.features.numpy(), '%f', delimiter=",")
        np.savetxt(top_dir + '/edge_list.edg', self.raw_edge_list.numpy().transpose(), '%d', delimiter=",")

    def __init__(self, edge_list: torch.Tensor, features: torch.Tensor, labels: torch.Tensor):
        """
        :param edge_list: 2 * M, Long
        :param features:  N * d, float
        :param labels:  N, Long
        :param split_setting: [train_each_class, valid]
        """

        self.dataname = ''

        self.raw_edge_list = edge_list
        self.__raw_adj = None

        self.features = features
        self.labels = labels

        self.num_features = features.size(1)
        self.num_classes = int(torch.max(labels)) + 1
        self.num_data = features.size(0)

        # initialize mask
        self.train_mask = None
        self.valid_mask = None
        self.tests_mask = None

        self.update_mask()
        self.num_train = torch.sum(self.train_mask.int(), dim=0).item()
        self.num_valid = torch.sum(self.valid_mask.int(), dim=0).item()
        self.num_tests = torch.sum(self.tests_mask.int(), dim=0).item()

    # def to(self, device):
    #     self.adj = self.adj.to(device)
    #     self.edge_list = self.edge_list.to(device)
    #     self.features = self.features.to(device)
    #     self.labels = self.labels.to(device)
    #     self.train_mask = self.train_mask.to(device)
    #     self.valid_mask = self.valid_mask.to(device)
    #     self.tests_mask = self.tests_mask.to(device)

    @property
    def sA(self):
        return self.raw_adj

    @property
    def raw_adj(self):
        # if self.__raw_adj is None:
        #     self.__raw_adj = edgelist2adj(self.raw_edge_list)
        # return self.__raw_adj
        return edgelist2adj(self.raw_edge_list)

    def update_mask(self, train_ratio=0.6, valid_ratio=0.2, seed=None):
        assert train_ratio + valid_ratio < 1, ('No test data!', train_ratio, valid_ratio)
        self.train_mask, self.valid_mask, self.tests_mask = split_by_ratio(self.labels, train_ratio, valid_ratio, seed)

        self.num_train = torch.sum(self.train_mask.int(), dim=0).item()
        self.num_valid = torch.sum(self.valid_mask.int(), dim=0).item()
        self.num_tests = torch.sum(self.tests_mask.int(), dim=0).item()

    def print_statisitcs(self):

        edge_list = self.raw_edge_list.numpy().transpose()

        G = nx.Graph()
        G.add_edges_from(edge_list)

        print("=== Dataset Statistics ===")
        print('data name  :', self.dataname)
        print('classes    :', self.num_classes)
        print("nodes      :", len(nx.nodes(G)))
        print("edges      :", len(nx.edges(G)))
        print('features   :', self.num_features)
        print("components :", nx.number_connected_components(G))

        percentage = label_percentage(self)
        print('class')
        print('  counts   :', label_counts(self))
        print('  ratio    :', ['%5.2f' % x for x in label_percentage(self)], '%')
        print('  majority :%6.2f %%' % max(percentage))
        print('  minority :%6.2f %%' % min(percentage))
        print('Homophily  :%6.2f %%' % (homophily(G, self.labels) * 100))
        print("==========================")

    def RVE(self, sample_size, seed=None):
        """
        Sampling graph
        :param sample_size:
        :return:
        """
        np.random.seed(seed)

        labels = self.labels.numpy()
        features = self.features.numpy()
        edge_list = self.raw_edge_list.numpy().transpose()

        G = nx.Graph()
        G.add_edges_from(edge_list)

        seed_node = int(random.choice(list(G.nodes)))  # np.random.permutation(np.array(G.nodes))[0]
        chosen_nodes_set = set()
        chosen_nodes_set.add(int(seed_node))
        neighbors_set = set(list(nx.neighbors(G, seed_node)))

        for i in range(1, sample_size):
            adding_node = int(random.choice(tuple(neighbors_set)))
            chosen_nodes_set.add(adding_node)
            neighbors_set = neighbors_set.union(set(nx.neighbors(G, adding_node))).difference(chosen_nodes_set)

        chosen_nodes = np.array(list(chosen_nodes_set))
        G = nx.subgraph(G, chosen_nodes)

        sampled_nodes = np.array(sorted(nx.nodes(G)), dtype=np.int)
        num_nodes = len(sampled_nodes)

        labels = labels[sampled_nodes]
        features = features[sampled_nodes]

        # mapping
        remap = {}
        for i in range(num_nodes):
            remap[sampled_nodes[i]] = i
        G = nx.relabel_nodes(G, mapping=remap)

        # oubling edge_list
        edge_list = np.array(nx.edges(G), dtype=np.int).transpose()  # 2,M
        directed = np.stack((edge_list[1], edge_list[0]), axis=0)
        edge_list = np.concatenate((edge_list, directed), axis=1)

        data = NodeData(torch.tensor(edge_list, dtype=torch.long), torch.tensor(features, dtype=torch.float),
                        torch.tensor(labels, dtype=torch.long))

        return data
