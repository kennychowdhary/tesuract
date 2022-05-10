import numpy as np
import pdb, warnings, pickle

def get_tuples(length, total):
    if length == 1:
        yield (total,)
        return

    for i in range(total + 1):
        for t in get_tuples(length - 1, total - i):
            yield (i,) + t


def multiindex2(dim, order):
    """Much faster total order construction"""
    L = []
    for o in range(order + 1):
        L = L + list([t[::-1] for t in get_tuples(dim, o)])
    return L


class RecursiveHypMultiIndex:
    """
    Adapted from MUQ (https://mituq.bitbucket.io/source/_site/index.html) but modified to
        be pure Python
    """

    def __init__(self, dim, order):
        self.maxOrder = order
        self.dim = dim
        self.minOrder = 0
        self.index = self.getIndex()

    def getIndex(self):
        output = []
        currDim = 0
        base = np.zeros(self.dim, dtype=int)
        # new for hyperbolic multiindex
        self.nugget = 1e-5
        q = 0.5
        maxNormPow = self.maxOrder**q + self.nugget
        output = self.RecHype(maxNormPow, output, currDim, base, q)
        return np.array(output, dtype=int)

    def RecHype(self, maxNormPow, output, currDim, base, q):
        total_order = MultiIndex(dim=self.dim, order=self.maxOrder)
        mindex = total_order.getIndex()
        mindex_new = mindex[
            [
                np.sum(m**q) ** (1.0 / q) < (self.maxOrder + self.nugget)
                for m in mindex
            ]
        ]
        return mindex_new


# multindex class
class MultiIndex:
    """
    Parts are adapted from UQTk (https://github.com/sandialabs/UQTk), but all code has been heavily modified to be pure Python.

    For total order multiindex construction
         Usage:
         M = MultiIndex(dim=2,order=2)
         print(M.getIndex())
         works with dim >= 1 and order >=1
    """

    def __init__(self, dim, order, mindex_type="total_order"):
        self.dim = dim
        self.order = order
        self.nPCTerms = self.computeNPCTerms()
        self.index_exists = False
        self.index = np.zeros((1, 1))
        if mindex_type == "total_order":
            self.getIndex()  # calculate multindex if not given as input
        if mindex_type == "hyperbolic":
            HmI = RecursiveHypMultiIndex(dim, order)
            self.index_exists = True  # index created flag
            self.index = HmI.index
            self.nPCTerms = len(HmI.index)

    def setIndex(self, mI):
        # to wrap custom array index as a index object
        self.index = mI
        self.nPCTerms = len(mI)

    def getIndex(self):
        if self.index_exists == False:
            # print('Index has not been set')
            # print('Setting to total order...')
            self.index = self.computeIndex()
            self.index_exists = True
        return self.index

    def computeNPCTerms(self):
        """Adapted form UQTk (https://github.com/sandialabs/UQTk)"""
        enume = 1
        denom = 1
        minNO = np.amin([self.order, self.dim])
        for k in range(minNO):
            enume *= self.order + self.dim - k
            denom *= k + 1
        nPCTerms = enume / denom
        return int(nPCTerms)

    def computeIndex(self):
        index = np.array(multiindex2(self.dim, self.order), dtype=int)
        return index
