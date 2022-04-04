def kmeans(X, n_clusters=9, random_state=0):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(X)

def dbscan(X, eps=10, min_samples=15):
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

def slink(X, n_clusters=9):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)

# Configs
DATA_SET = '../data/clusterData1/clusterData1.10k.dat'
RESULT_DIR = './result/'
METHOD = slink

import numpy as np
import matplotlib.pyplot as plt

with open(DATA_SET, 'r') as f:
    X = []
    for line in f:
        x, y = map(float, line.split())
        X.append([x, y])
    X = np.array(X)
    # plt.scatter(X[:, 0], X[:, 1], marker='o')
    # plt.savefig(RESULT_DIR + 'points.png')

    y_pred = METHOD(X)
    plt.cla()
    plt.clf()
    plt.figure(figsize=(10,8))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.savefig(RESULT_DIR + 'result_' + METHOD.__name__ + '.png', dpi=100)
