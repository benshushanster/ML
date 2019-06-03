import numpy as np

from decision_tree import *

__all__ = ['RandomForest']


class RandomForest:
    def __init__(self, forest_size=5, max_depth=3, use_gini=True, sub_sample_pcnt=1):
        self._forestRoot = []
        self._max_depth = max_depth
        self._use_gini = use_gini
        self._forest_size = forest_size
        self._sub_sample_pcnt = sub_sample_pcnt

    def _create_subsample(self, n):
        if self._sub_sample_pcnt > 1:
            self._sub_sample_pcnt = 1
        pos = np.random.randint(0, n, n * self._sub_sample_pcnt)
        return pos

    def train(self, x_train, y_train, ):
        for i in range(self._forest_size):
            samples = self._create_subsample(len(x_train))
            tree = DecisionTree()
            tree.train(x_train[samples], y_train[samples], self._max_depth, self._use_gini, random_features=True)
            self._forestRoot.append(tree)

    def test(self, x_test, y_test):
        correct = 0
        for i in range(x_test.shape[0]):
            if y_test[i] == self.predict(x_test[i]):
                correct += 1
        return correct / x_test.shape[0] * 100

    def predict(self, data):
        res = 0
        for tree in self._forestRoot:
            res += tree.predict(data)
        return res >= self._forest_size
