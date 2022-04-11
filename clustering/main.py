# ---------------------- Configs ----------------------
USE_MY = True
DATA_SETS = [
        './data/clusterData1/clusterData1.10k.dat',
        './data/clusterData1/clusterData2.8k.dat',
        './data/clusterData1/clusterData3.8k.dat',
        './data/clusterData1/clusterData4.8k.dat',
    ]
PARAMS = [
    # KMeans, SLink, DBScan
    [(9,), (9,), (10, 15)],
    [(6,), (6,), (10, 18)],
    [(6,), (6,), (8, 16)],
    [(7,), (7,), (11.5, 8)],
]
RESULT_DIR = './result/'
# -----------------------------------------------------

if USE_MY:
    from kmeans import KMeans
    from dbscan import DBSCAN
    from slink import AgglomerativeClustering
    RESULT_DIR += 'my/'
else:
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    RESULT_DIR += 'sklearn/'

def kmeans(X, n_clusters=9, random_state=0):
    return KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(X)

def slink(X, n_clusters=9):
    return AgglomerativeClustering(n_clusters=n_clusters, linkage='single').fit_predict(X)

def dbscan(X, eps=10, min_samples=15):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

METHODS = [kmeans, slink, dbscan]

# -----------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import timeit
import json

all_logs = {}
for i, data_set in enumerate(DATA_SETS):
    with open(data_set, 'r') as f:
        X = []
        for line in f:
            x, y = map(float, line.split())
            X.append([x, y])
        X = np.array(X)
        # plt.scatter(X[:, 0], X[:, 1], marker='o')
        # plt.savefig(RESULT_DIR + 'points.png')
        for j, method in enumerate(METHODS):
            logs = {}
            run_numbers = 20
            def target():
                global y_pred
                y_pred = method(X, *PARAMS[i][j])
            time = timeit.timeit(target, number=run_numbers, globals=locals())
            logs['data_set'] = data_set.split('/')[-1]
            logs['USE_MY'] = USE_MY
            logs['avg_run_time'] = time / run_numbers
            plt.cla()
            plt.clf()
            plt.figure(figsize=(5,4))
            plt.title('\"' + method.__name__ + '\"' + ' clustering results on '+ 'Cluster Data ' + str(i+1))
            plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10)
            file_name = RESULT_DIR + 'result' + str(i+1) + '_' + method.__name__
            plt.savefig(file_name + '.png', dpi=200)
            print(method.__name__, ':', logs)
            all_logs.setdefault(method.__name__, [])
            all_logs[method.__name__].append(logs)
json.dump(all_logs, open(RESULT_DIR + 'result.json', 'w+'))
