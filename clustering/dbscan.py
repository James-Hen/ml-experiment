import numpy as np
from scipy import spatial

UNCLASSIFIED = False
NOISE = False

class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def _eps_neighborhood(self, p, eps):
        return self.tree.query_ball_point(p, eps)

    def _expand_cluster(self, classifications, point_id, cluster_id, eps, min_samples):
        seeds = self._eps_neighborhood(self.tree.data[point_id], eps)
        if len(seeds) < min_samples:
            classifications[point_id] = NOISE
            return False
        else:
            classifications[point_id] = cluster_id
            for seed_id in seeds:
                classifications[seed_id] = cluster_id
                
            while len(seeds) > 0:
                current_point = seeds[0]
                results = self._eps_neighborhood(self.tree.data[current_point], eps)
                if len(results) >= min_samples:
                    for i in range(0, len(results)):
                        result_point = results[i]
                        if classifications[result_point] == UNCLASSIFIED or \
                        classifications[result_point] == NOISE:
                            if classifications[result_point] == UNCLASSIFIED:
                                seeds.append(result_point)
                            classifications[result_point] = cluster_id
                seeds = seeds[1:]
            return True

    def fit_predict(self, m):
        self.tree = spatial.KDTree(m)
        cluster_id = 1
        n_points = len(m)
        classifications = [UNCLASSIFIED] * n_points
        for point_id in range(0, n_points):
            if classifications[point_id] == UNCLASSIFIED:
                if self._expand_cluster(classifications, point_id, cluster_id, self.eps, self.min_samples):
                    cluster_id = cluster_id + 1
        return classifications