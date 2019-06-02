import numpy as np

__all__ = ['RandomForest']


class RandomForest:
    def __init__(self):
        self._root = None

    def _create_subsample(self, data, pcnt=1):
        n = len(data)
        if pcnt > 1:
            pcnt = 1
        pos = np.random.randint(0, n, n * pcnt)
        return data[pos]

    def train(self, x_train, y_train, max_depth=3, use_gini=True):
        pass

    def test(self, x_test, y_test):
        correct = 0
        for i in range(x_test.shape[0]):
            if y_test[i] == self.predict(x_test[i]):
                correct += 1
        return correct / x_test.shape[0] * 100

    def predict(self, data):
        return self._root.predict(data)
