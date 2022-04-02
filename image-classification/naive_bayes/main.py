# Variables
DATASET_PATH = '../data/'
RESULT_PATH = './result/'

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torchvision import datasets

trainset = datasets.FashionMNIST(DATASET_PATH, download = True, train = True)
testset = datasets.FashionMNIST(DATASET_PATH, download = True, train = False)
x_train = trainset.data.numpy()
x_test = testset.data.numpy()
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2])) > 127
y_train = trainset.targets.numpy()
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2])) > 127
y_test = testset.targets.numpy()

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
model2 = BernoulliNB()
model2.fit(X_train, y_train)


pred = model2.predict(X_test)
score = accuracy_score(y_test, pred)
print('Score:', score)
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