import numpy as np
from scipy.spatial import Delaunay

class UnionFindSet:
    def __init__(self, n):
        self.root = list(range(n))
    def find(self, a):
        root = a
        while self.root[root] != root:
            root = self.root[root]
        while self.root[a] != a:
            a, self.root[a] = self.root[a], root
        return root
    def union(self, a, b):
        x = self.find(a)
        y = self.find(b)
        if x != y:
            self.root[y] = x
            return True
        return False

class AgglomerativeClustering:
    def __init__(self, n_clusters, linkage='single', max_dist=10):
        self.n_clusters = n_clusters
        self.max_dist = max_dist
    def fit_predict(self, data):
        n = len(data)
        self.d = Delaunay(data)
        links = np.array([[[a, b], [a, c], [b, c]] for a, b, c, in self.d.simplices]) \
            .reshape(len(self.d.simplices) * 3, 2)
        dists = np.array([np.linalg.norm(self.d.points[l[0]] - self.d.points[l[1]]) for l in links])
        srt_dists = sorted(zip(links, dists), key=lambda x:x[-1])
        ufs = UnionFindSet(n)
        cur_clusters = n
        for (u, v), _dist in srt_dists:
            if cur_clusters <= self.n_clusters:
                break
            if ufs.union(u, v):
                cur_clusters -= 1
        return ufs.root
