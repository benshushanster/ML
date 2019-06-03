import pandas as pd
from sklearn.model_selection import train_test_split

from random_forest import *

df = pd.read_csv('./wdbc.data', header=None)
XX = (df.iloc[:, 2:]).values
y = (df.iloc[:, 1] == 'M').values

X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=0.30, random_state=70, shuffle=True)
rf = RandomForest()
rf.train(X_train, y_train)
print("Accuracy:{0:.2f}%".format(rf.test(X_test, y_test)))
