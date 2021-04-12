import torch


def is_sparse(x: torch.Tensor) -> bool:
    """
    :param x:
    :return: True if x is sparse tensor else False
    """
    try:
        x._indices()
    except RuntimeError:
        return False
    return True


class SparseTensor(torch.Tensor):
    """
    NeverUse
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError()

    def _indices(self):
        raise NotImplementedError()

    def _values(self):
        raise NotImplementedError()

    def to_dense(self):
        raise NotImplementedError()


def sparse_discard_zeros(sX: SparseTensor) -> SparseTensor:
    i = sX._indices()
    v = sX._values()
    nonzero = torch.where(v != 0)[0]
    i = i[:, nonzero]
    v = v[nonzero]
    return torch.sparse.FloatTensor(i, v)

# def sparse_dense_mul(sX: SparseTensor, Y: torch.Tensor, discard=False) -> SparseTensor:
#     i = sX._indices()
#     v = sX._values()
#     dv = Y[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
#     v = v * dv
#
#     if discard:
#         # discard 0
#         nonzero = torch.where(v != 0)[0]
#         i = i[:, nonzero]
#         v = v[nonzero]
#     return torch.sparse.FloatTensor(i, v)
#
#
# def sparse_dense_matmul(sX: SparseTensor, Y: torch.Tensor) -> torch.Tensor:
#     return torch.sparse.mm(sX, Y)
