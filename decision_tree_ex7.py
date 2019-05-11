import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DecisionTree:
    def __init__(self):
        self._root = None

    def train(self, x_train, y_train, max_depth=3):
        data = np.column_stack([x_train, y_train])
        self._root = Node(data, 0, max_depth)

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
    def split(data, idx, value, remove_feature_after_split=False):
        predicate = data[:, idx] > value
        group_a = data[np.where(predicate)]
        group_b = data[np.where(np.logical_not(predicate))]
        if remove_feature_after_split:
            return np.delete(group_a, idx, axis=1), np.delete(group_b, idx, axis=1)
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
    def gain(entropy, group_a, group_b):
        # calculating the total items in the both groups
        total = group_a.shape[0] + group_b.shape[0]
        # calculate entropy for each group
        entropy_a = Node.entropy(Node.probability(group_a))
        entropy_b = Node.entropy(Node.probability(group_b))
        # calculate the information gain
        return entropy - group_a.shape[0] / total * entropy_a - group_b.shape[0] / total * entropy_b

    @staticmethod
    def get_split_by_entropy(data):
        entropy = Node.entropy(Node.probability(data))
        max_gain = 0
        res = (None, None)
        for featureIdx in range(data.shape[1] - 1):
            for rowIdx in range(data.shape[0]):
                value = data[rowIdx, featureIdx]
                group_a, group_b = Node.split(data, featureIdx, value)
                gain = Node.gain(entropy, group_a, group_b)
                if gain > max_gain:
                    max_gain = gain
                    res = (featureIdx, value)
        return res

    def __init__(self, data, depth, max_depth):
        self.max_depth = max_depth
        self.depth = depth
        self.data = data
        self.predicted = Node.probability(data) > 0.5
        self.left = None
        self.right = None
        self.featureIdx = None
        self.splitValue = None
        self._split_nodes()

    def _split_nodes(self):
        # if reached to max depth exit
        if self.max_depth < self.depth:
            return
        # if only one item exit
        if self.data.shape[0] == 1:
            return
        # if no features left exit
        if self.data.shape[1] == 1:
            return

        self.featureIdx, self.splitValue = Node.get_split_by_entropy(self.data)
        if (self.featureIdx is None) or (self.splitValue is None):
            return
        group_a, group_b = Node.split(self.data, self.featureIdx, self.splitValue, remove_feature_after_split=True)

        if group_a.shape[0] > 0:
            self.left = Node(group_a, self.depth + 1, self.max_depth)
        if group_b.shape[0] > 0:
            self.right = Node(group_b, self.depth + 1, self.max_depth)

    def _is_leaf(self):
        return self.left is None and self.right is None

    def predict(self, data):
        if self._is_leaf():
            return self.predicted
        if data[self.featureIdx] > self.splitValue:
            return self.left.predict(np.delete(data, self.featureIdx))
        return self.right.predict(np.delete(data, self.featureIdx))


df = pd.read_csv('./wdbc.data', header=None)
XX = (df.iloc[:, 2:]).values
y = (df.iloc[:, 1] == 'M').values

X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=0.20, random_state=90, shuffle=True)
dt = DecisionTree()
dt.train(X_train, y_train, max_depth=4)
print("Accuracy:{0:.2f}%".format(dt.test(X_test, y_test)))

