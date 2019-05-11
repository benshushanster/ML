import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, :2]  # we only take the first two features.
y = np.array(iris.target_names[iris.target] == 'setosa', dtype='int')

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr_r")
plt.show()


# plt.close()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def h(x, t):
    return sigmoid(x @ t)


def loss(x, y, t):
    m = x.shape[0]
    res = 1 / m * (np.dot(-y, np.log(h(x, t))) - np.dot(1 - y, np.log(1 - h(x, t))))
    return res


def grads(x, y, t):
    m = x.shape[0]
    return 1 / m * ((h(x, t) - y) @ x)


def train(x, y, t, max_epoc=2000, alpha=1):
    theta = t
    for epoch in range(max_epoc):
        l = loss(x, y, t)
        grad = grads(x, y, t)
        theta -= alpha * grad
        print("epoch:{} loss:{}".format(epoch, l))
    return theta


XX = np.hstack((np.ones((X.shape[0], 1)), X))
Theta = np.array([2.0, 2.0, 0.0])

model = train(XX, y, np.array([1.3,2,1.2]) ,150000,5)
my_x = np.linspace(3.5, 8.0, 40)
my_y = (model[0] + my_x * model[1]) / model[1]
plt.plot(my_x, my_y)
