import random

import numpy as np

__all__ = ['DecisionTree']


class DecisionTree:
    def __init__(self):
        self._root = None

    def train(self, x_train, y_train, max_depth=3, use_gini=True, random_features=False):
        data = np.column_stack([x_train, y_train])
        self._root = Node(data, 0, max_depth, use_gini, random_features)

    def test(self, x_test, y_test):
        correct = 0
        for i in range(x_test.shape[0]):
            if y_test[i] == self.predict(x_test[i]):
                correct += 1
        return correct / x_test.shape[0] * 100

    def predict(self, data):
        return self._root.predict(data)


class Node:

    @staticmethod
    def split(data, idx, value):
        predicate = data[:, idx] > value
        group_a = data[np.where(predicate)]
        group_b = data[np.where(np.logical_not(predicate))]
        return group_a, group_b

    @staticmethod
    def entropy(p):
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    @staticmethod
    def probability(data):
        if data.shape[0] == 0:
            return 0
        positives = data[:, -1].sum()
        return positives / data.shape[0]

    @staticmethod
    def gini(p):
        return 1 - (p ** 2 + (1 - p) ** 2)

    @staticmethod
    def avg_gini(group_a, group_b):
        total = group_a.shape[0] + group_b.shape[0]
        # calculate entropy for each group
        gini_a = Node.gini(Node.probability(group_a))
        gini_b = Node.gini(Node.probability(group_b))
        return group_a.shape[0] / total * gini_a + group_b.shape[0] / total * gini_b

    @staticmethod
    def gain(entropy, group_a, group_b):
        # calculating the total items in the both groups
        total = group_a.shape[0] + group_b.shape[0]
        # calculate entropy for each group
        entropy_a = Node.entropy(Node.probability(group_a))
        entropy_b = Node.entropy(Node.probability(group_b))
        # calculate the information gain
        return entropy - group_a.shape[0] / total * entropy_a - group_b.shape[0] / total * entropy_b

    @staticmethod
    def get_split_by_entropy(data, random_features):
        entropy = Node.entropy(Node.probability(data))
        max_gain = 0
        res = (None, None)
        features = range(data.shape[1] - 1)
        if random_features:
            # random samples features sqrt of th total size
            features = random.sample(features, int(np.sqrt(data.shape[1] - 1)))

        for featureIdx in features:
            for rowIdx in range(data.shape[0]):
                value = data[rowIdx, featureIdx]
                group_a, group_b = Node.split(data, featureIdx, value)
                gain = Node.gain(entropy, group_a, group_b)
                if gain > max_gain:
                    max_gain = gain
                    res = (featureIdx, value)
        return res

    @staticmethod
    def get_split_by_gini(data, random_features):
        min_gini = 1
        res = (None, None)
        features = range(data.shape[1] - 1)
        if random_features:
            # random samples features sqrt of th total size
            features = random.sample(features, int(np.sqrt(data.shape[1] - 1)))
        for featureIdx in features:
            for rowIdx in range(data.shape[0]):
                value = data[rowIdx, featureIdx]
                group_a, group_b = Node.split(data, featureIdx, value)
                gini = Node.avg_gini(group_a, group_b)
                if gini <= min_gini:
                    min_gini = gini
                    res = (featureIdx, value)
        return res

    def __init__(self, data, depth, max_depth, use_gini, random_features):
        self.max_depth = max_depth
        self.depth = depth
        self.data = data
        self.predicted = Node.probability(data) > 0.5
        self.left = None
        self.right = None
        self.featureIdx = None
        self.splitValue = None
        self.use_gini = use_gini
        self.random_features = random_features
        self._split_nodes()

    def _split_nodes(self):
        # if reached to max depth exit
        if self.max_depth < self.depth:
            return
        # if only one item exit
        if self.data.shape[0] == 1:
            return

        if self.use_gini:
            self.featureIdx, self.splitValue = Node.get_split_by_gini(self.data, self.random_features)
        else:
            self.featureIdx, self.splitValue = Node.get_split_by_entropy(self.data, self.random_features)

        if (self.featureIdx is None) or (self.splitValue is None):
            return
        group_a, group_b = Node.split(self.data, self.featureIdx, self.splitValue)

        if group_a.shape[0] > 0:
            self.left = Node(group_a, self.depth + 1, self.max_depth, self.use_gini, self.random_features)
        if group_b.shape[0] > 0:
            self.right = Node(group_b, self.depth + 1, self.max_depth, self.use_gini, self.random_features)

    def _is_leaf(self):
        return self.left is None and self.right is None

    def predict(self, data):
        if self._is_leaf():
            return self.predicted
        if data[self.featureIdx] > self.splitValue:
            if self.left is None:
                return self.predicted
            return self.left.predict(data)
        if self.right is None:
            return self.predicted
        return self.right.predict(data)
