import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

class KMeans:
    def __init__(self, n_clusters, random_state=0, verbose=False, tol=1e-4, max_iter=10):
        self.k = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.rs = RandomState(MT19937(SeedSequence(random_state)))

    def fit_predict(self, data):
        self.centers = data[self.rs.choice(range(len(data)), self.k, replace=False)]
        for epoch in range(self.max_iter):
            next_centers = np.zeros((self.k, *data.shape[1:]))
            pop_count = np.zeros((self.k,))
            labels = [np.argmin(np.linalg.norm(self.centers - p, axis=-1)) for p in data]
            for l, p in zip(labels, data):
                next_centers[l] += p
                pop_count[l] += 1
            next_centers /= pop_count[np.newaxis].T
            diff = np.average(np.linalg.norm(self.centers - next_centers, axis=-1))

            if (self.verbose):
                print("Epoch:", epoch, "; diff:", diff, "; centers:", pop_count)

            if (diff < self.tol):
                break
            self.centers = next_centers

        return labels

    def predict(self, p):
        return np.argmin(np.linalg.norm(self.centers - p))