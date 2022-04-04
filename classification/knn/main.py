# Variables
DATASET_PATH = '../data/'
RESULT_PATH = './result_full/'

from numba import njit
import numpy as np
import operator 
from operator import itemgetter

def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def xor_dist(x1, x2):
    return sum(x1 ^ x2)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torchvision import datasets

trainset = datasets.FashionMNIST(DATASET_PATH, download = True, train = True)
testset = datasets.FashionMNIST(DATASET_PATH, download = True, train = False)
x_train = trainset.data.numpy()
x_test = testset.data.numpy()
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))# > 127
y_train = trainset.targets.numpy()
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))# > 127
y_test = testset.targets.numpy()

print('Data type: ', X_train.dtype)
print('Trainning set shapes: ', X_train.shape, y_train.shape)
print('Testing set shapes:   ', X_test.shape, y_test.shape)
print('Trainning label sizes:', np.unique(y_train,return_counts=True))
print('Testing label sizes:  ', np.unique(y_test,return_counts=True))

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

def predict(dist_func, K, X_test):
    global X_train, y_train
    predictions = [] 
    for i in range(len(X_test)):
        dist = np.array([dist_func(X_test[i], x_t) for x_t in X_train])
        dist = dist.argpartition(K)[:K]
        neigh_count = {}
        for idx in dist:
            if y_train[idx] in neigh_count:
                neigh_count[y_train[idx]] += 1
            else:
                neigh_count[y_train[idx]] = 1
        sorted_neigh_count = sorted(neigh_count.items(), key=operator.itemgetter(1), reverse=True)
        predictions.append(sorted_neigh_count[0][0])
    return predictions

from multiprocessing import Pool, current_process
import os

def test_acc_for(k):
    print("Doing for ", k)
    pred = predict(euc_dist, k, X_test[:1000])
    acc = accuracy_score(y_test[:1000], pred)
    print("K = " + str(k) + "; Accuracy: " + str(acc))
    return acc

if __name__ ==  '__main__':
    kVals = np.arange(3,100,2)
    max_cpu = os.cpu_count()
    with Pool(max_cpu) as p:
        accuracies = p.map(test_acc_for, kVals)

    max_index = accuracies.index(max(accuracies))
    print(max_index)

    from matplotlib import pyplot as plt 

    plt.figure(figsize=(10,8))
    plt.plot(kVals, accuracies)
    plt.xlabel("K Value") 
    plt.ylabel("Accuracy")
    plt.savefig("AccuracywithK.png", dpi=100)
    import time
    timer_start = time.process_time_ns()
    pred = predict(euc_dist, kVals[max_index], X_test)
    print("Run time", time.process_time_ns() - timer_start)
    score = accuracy_score(y_test, pred)
    print(score)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, pred)
    
    classes = [
            'T-shirt/top',
            'Trouser',
            'Pullover',
            'Dress',
            'Coat',
            'Sandal',
            'Shirt',
            'Sneaker',
            'Bag',
            'Ankle boot',
        ]

    print("Precision \n", precision)
    print("\nRecall \n", recall)
    print("\nF-score \n", fscore)

    print('\nMeasures', end=' ')
    for c in classes:
        print('&', c, end=' ')
    print('\\\\\nPrecision', end=' ')
    for p in precision:
        print('& {:.3f}'.format(p), end=' ')
    print('\\\\\nRecall', end=' ')
    for r in recall:
        print('& {:.3f}'.format(r), end=' ')
    print('\\\\\nF-score', end=' ')
    for f in fscore:
        print('& {:.3f}'.format(f), end=' ')
    print('\\\\')