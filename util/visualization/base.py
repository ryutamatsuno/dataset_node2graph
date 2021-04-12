import matplotlib
import matplotlib.patches as pat
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from umap import UMAP


def get_colors(num_colors: int) -> []:
    # generate colors from cmap
    assert num_colors > 0
    if num_colors == 1:
        return ['#009191']
    if num_colors < 8:
        # return ['red', 'blue', 'green', 'grey', 'aqua', 'orange', 'purple', 'sage'][:num_colors]
        return ['#000091', '#f54848', '#009191', '#f59148', '#919191', '#F591F5', '#48F5F5', '#F5F500'][:num_colors]

    cmap = matplotlib.cm.get_cmap('jet')
    # cmap = matplotlib.cm.get_cmap('hsv')
    colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
    return colors


def get_markers(num_marks: int) -> []:
    # ".",
    return ["o", "D", ",", "^", "v", "*", "p", "<", ">", "1", "2", "3", "4", "8", "s", "h", "H", "+", "x", "d", "|", "_", ][:num_marks]


def hid22D(data, method='UMAP'):
    if type(data) is list:
        data = np.array(data)
    elif type(data) is torch.Tensor:
        data = data.detach().numpy()
    dim_x = data.shape[1]

    if data.shape[1] == 2:
        X_reduced = data
    elif method == 'PCA':
        X_reduced = PCA(n_components=2).fit_transform(data)
    elif method == 'TSNE':
        X_reduced = TSNE(n_components=2).fit_transform(data)
    elif method == 'PCA+TSNE':
        if dim_x > 10:
            data = PCA(n_components=10).fit_transform(data)
        X_reduced = TSNE(n_components=2).fit_transform(data)
    elif method == 'UMAP':
        # X_reduced = UMAP(n_components=2).fit_transform(data)
        X_reduced = UMAP(n_components=2, min_dist=0.2, n_neighbors=10).fit_transform(data)
    elif method == 'PCA+UMAP':
        if dim_x > 10:
            data = PCA(n_components=10).fit_transform(data)
        # X_reduced = UMAP(n_components=2).fit_transform(data)
        X_reduced = UMAP(n_components=2, min_dist=0.2, n_neighbors=10).fit_transform(data)

    # elif method == 'LDA':
    #     # normalization
    #     u = np.mean(data, axis=0)
    #     s = np.std(data - u, axis=0) + 1e-10
    #     data = (data - u) / s
    #     # clf = LDA(n_components=2)
    #     # clf.fit(data,labels)
    #     # X_reduced = clf.transform(data)
    #     X_reduced = LDA(n_components=2).fit_transform(data, labels)
    #     # print(X_reduced.shape)
    else:
        raise ValueError('method error')
    return X_reduced
