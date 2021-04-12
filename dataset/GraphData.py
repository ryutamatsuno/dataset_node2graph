import glob
import os
import time

import networkx as nx
import numpy as np
import torch

from util import SparseTensor

TOPDIR = 'dataset/'
import os
import numpy as np
from tqdm import tqdm


def double_edges(E: torch.Tensor):
    reversed = torch.stack((E[1], E[0]), dim=0)
    E = torch.cat((E, reversed), dim=1)
    return E


def E2sA(E: torch.Tensor) -> SparseTensor:
    weights = torch.ones(E.size(1))
    sA = torch.sparse.FloatTensor(E, weights)
    return sA


def read_edge_file(filepath) -> torch.Tensor:
    with open(filepath, 'r') as f:
        lines = f.readlines()
    edges = []
    for l in lines:
        if ',' in l:
            s = l.split(',')
        else:
            s = l.split(' ')
        u = int(s[0])
        v = int(s[1])
        # remove self-loops
        if u == v:
            continue
        e = (u, v)
        edges.append(e)

    edgelist = torch.tensor(edges, dtype=torch.long).transpose(0, 1)
    assert edgelist.shape[0] == 2
    return edgelist  # 2, M


class GraphData(object):

    @staticmethod
    def load(data_name):
        start = time.time()

        dir_path = TOPDIR + data_name

        Es = []  # graph
        xs = []  # node feature
        labels = []  # graph labels

        # label
        if os.path.exists(dir_path + '/0'):
            # labels are integer
            num_labels = len(os.listdir(dir_path))
            d_names = [str(i) for i in range(num_labels)]
        else:
            d_names = sorted(os.listdir(dir_path))

        for label, d_name in enumerate(d_names):
            d_path = dir_path + '/' + d_name
            paths = sorted(glob.glob(d_path + '/*.edg'))

            for edge_path in paths:
                und_edges = read_edge_file(edge_path)
                E = double_edges(und_edges)

                Es.append(E)
                labels.append(label)

        labels = torch.tensor(labels, dtype=torch.long)
        print('load Data done:', time.time() - start, '[sec]')
        return GraphData(Es, labels, xs)

    @property
    def Gs(self):
        if self.__Gs is not None:
            return self.__Gs
        Gs = []
        for i in range(self.num_data):
            g = nx.Graph()
            g.add_edges_from(torch.transpose(self.Es[i], 0, 1).numpy())
            Gs.append(g)
        self.__Gs = Gs
        return self.__Gs

    @property
    def Ns(self):
        if self.__Ns is not None:
            return self.__Ns
        Ns = [self.sAs[i].shape[0] for i in range(self.num_data)]
        self.__Ns = torch.tensor(Ns, dtype=torch.long)
        return self.__Ns

    @property
    def max_nodes(self):
        return torch.max(self.Ns).item()

    def __init__(self, Es: [torch.Tensor], labels: torch.Tensor, xs: [torch.Tensor]):  # xs:[torch.Tensor],
        super().__init__()

        self.Es = Es
        self.xs = xs
        self.labels = labels  # num_data

        # assert len(Es) == labels.shape[0]
        self.num_classes = int(torch.max(labels)) + 1
        self.num_data = len(Es)

        # SparceTensors
        sAs = []
        for i in range(self.num_data):
            sA = E2sA(Es[i])
            sAs.append(sA)
        self.sAs = sAs

        # EdgeLists
        self.ELs = [E.transpose(0, 1) for E in Es]

        self.__Ns = None
        self.__Gs = None

    def save(self, name, flag_feat=False):

        os.mkdir(TOPDIR + name)
        for i in range(self.num_classes):
            os.mkdir(TOPDIR + name + '/%d' % i)

        for i in tqdm(range(self.num_data)):
            E = self.Es[i]
            feat = self.xs[i]
            label = int(self.labels[i])
            dir_path = TOPDIR + name + '/%d' % label + '/'

            # save edge
            np.savetxt(dir_path + '%d.edg' % i, E.numpy().transpose(), fmt='%d', delimiter=' ')
            if flag_feat:
                np.savetxt(dir_path + '%d.csv' % i, feat.numpy(), delimiter=',')
        print('save done')

    def print_statistics(self):

        print(" - statistics - ")
        print("graphs  :", self.num_data)
        print('classes :', self.num_classes)

        Ns = []
        Ms = []
        density = []
        for i in range(self.num_data):
            sA = self.sAs[i]
            E = self.Es[i]
            N = sA.shape[0]
            M = E.shape[1] / 2
            Ns.append(N)
            Ms.append(M)
            density.append(M / (N * (N - 1) / 2))

        print("max N:", max(Ns))
        print("min N:", min(Ns))
        print("ave N:", sum(Ns) / self.num_data)

        print("max M:", max(Ms))
        print("min M:", min(Ms))
        print("ave M:", sum(Ms) / self.num_data)

        print("ave density:", sum(density) / self.num_data)

        # counts for each label
        def label_counts(data: GraphData, mask=None) -> np.ndarray:
            counts = np.zeros(data.num_classes, dtype=np.int)
            labels = data.labels.numpy()
            for i in range(labels.shape[0]):
                if mask is not None:
                    if not mask[i]:
                        continue
                counts[int(labels[i])] += 1
            return counts

        def label_percentage(data: GraphData, mask=None) -> np.ndarray:
            counts = label_counts(data, mask).astype(np.float)
            counts = counts * 100 / np.sum(counts)
            return counts

        print('data label balance:', label_counts(self), ' = ', label_percentage(self), '%')

        percentage = label_percentage(self)
        print('majority:', max(percentage))
        print('minority:', min(percentage))
        print(" - ")
