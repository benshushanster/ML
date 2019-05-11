import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class NaiveBaseClassifier:
    def __init__(self):
        self._means = []
        self._vars = []

    def train(self, train_set):
        data = pd.DataFrame(train_set)
        self._means = data.groupby(0).mean()
        self._vars = data.groupby(0).var()

    def test(self, test_set):
        data = np.array(test_set)
        len = data.shape[0]
        if len == 0:
            return 0
        res = []
        for sample in data:
            res.append(self.predict(sample[1:]))
        return (data[:, 0] == res).sum() / len

    def predict(self, sample):
        return ((1 / np.sqrt(2 * np.pi * self._vars)) * np.exp(
            -np.square(sample - self._means) / (2 * self._vars))).product(axis=1).idxmax()


# prepare data
df = pd.read_csv('./pima-indians-diabetes.csv', header=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=90, shuffle=True)
test_set = np.column_stack([y_test, X_test])
train_set = np.column_stack([y_train, X_train])

# tarin and print Accuracy
nbc = NaiveBaseClassifier()
nbc.train(train_set)
print("Accuracy:{0:.2f}".format(nbc.test(test_set)))
