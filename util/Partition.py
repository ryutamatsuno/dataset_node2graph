import numpy as np


def is_partition_equal(pt1: [set()], pt2: [set()]):
    if len(pt1) != len(pt2):
        return False
    N = len(pt1)

    mask_pt2 = [False for _ in range(N)]

    for p in pt1:
        matched = False
        for j in range(N):
            if mask_pt2[j]:
                continue
            q = pt2[j]

            if p == q:
                matched = True
                mask_pt2[j] = True
                break

        if not matched:
            return False

    for m in mask_pt2:
        assert m
    return True


def compute_partiton(universe: list, indicator: [int]):
    assert isinstance(universe, list)
    assert isinstance(indicator, list), (type(indicator), indicator)
    assert isinstance(indicator[0], int)
    assert len(universe) == len(indicator)

    keys, inverse = np.unique(np.array(indicator, dtype=np.int), return_inverse=True)
    N_pt = len(keys)
    pt = [[] for _ in range(N_pt)]
    for i in range(len(universe)):
        element = universe[i]
        pt_i = inverse[i]
        pt[pt_i].append(element)
    pt = [set(x) for x in pt]
    return pt


class Partition():

    def __init__(self, universe, initialize='discrete'):
        super().__init__()

        assert isinstance(universe, set)

        self.__universe = universe

        if initialize == 'discrete':
            self.__partiton = [{s} for s in universe]
        elif initialize == 'trivial':
            self.__partiton = [universe.copy()]
        elif isinstance(initialize, list):
            self.__partiton = initialize.copy()
        elif isinstance(initialize, set):
            self.__partiton = list(initialize)
        else:
            raise ValueError(initialize)

        self.is_valid()

    def number_of_groups(self):
        return len(self.__partiton)

    def is_valid(self):
        pt = self.get_partion()

        u = set()
        for p in pt:
            u = u.union(p)
        # is pt covers the universe?
        assert u == self.__universe, ("universe does not match:", u, self.__universe)

        # is there no duplication?
        n_elements = sum([len(p) for p in pt])
        assert n_elements == len(u), ("duplicated elements:", n_elements, len(u), self.__partiton)

        # OK
        return True

    def get_group_of(self, x):
        ix = self.__get_index(x)
        if ix < 0:
            return None
        return self.__partiton[ix]

    def __get_index(self, x):
        if not x in self.__universe:
            raise ValueError()

        for k in range(len(self.__partiton)):
            if x in self.__partiton[k]:
                return k
        return -1

    def is_in_the_same_group(self, x, y):
        ix = self.__get_index(x)
        iy = self.__get_index(y)

        if ix < 0 or iy < 0:
            return False
        return ix == iy

    def merge_gropus_of(self, x, y):
        if x == y:
            raise ValueError()
        if not (x in self.__universe and y in self.__universe):
            raise ValueError()

        ix = self.__get_index(x)
        iy = self.__get_index(y)
        if ix == iy:
            return

        lp = len(self.__partiton)
        px: set = self.__partiton[ix]
        py: set = self.__partiton[iy]
        self.__partiton.remove(px)
        self.__partiton.remove(py)
        assert len(self.__partiton) == lp - 2

        self.__partiton.append(px.union(py))
        self.is_valid()

    def get_partion(self):
        pt = sorted(self.__partiton.copy(), key=lambda x: min(x))
        pt = sorted(pt, key=lambda x: -len(x))
        return pt

    def __eq__(self, other):
        if not self.__universe == other.__universe:
            return False
        if not is_partition_equal(self.__partiton, other.__partition):
            return False
        return True
