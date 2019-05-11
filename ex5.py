import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

all = np.column_stack([X, y])
np.random.shuffle(all)

plt.scatter(all[:, 0], all[:, 1], c=all[:, 2])
plt.show()


def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def search(a, k):
    res = []
    i = 0
    for item in all:
        res.append([distance(item[:2], a), i])
        i += 1
    arr = np.array(res)
    index = np.sort(arr.view('i8,i8'), order=['f0'], axis=0).view(np.float)[:k, 1]
    index = index.astype(int)
    return index


def predict(a, k=3):
    index = search(a, k)
    cat = all[index]
    cat = cat[:, 2]
    return np.max(np.column_stack(np.unique(cat, return_counts=True)), axis=0)[0]


X = np.linspace(0,8,100)
Y = X.copy()
X, Y = np.meshgrid(X, Y)

