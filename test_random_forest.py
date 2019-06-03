import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from random_forest import *

df = pd.read_csv('./wdbc.data', header=None)
XX = (df.iloc[:, 2:]).values
y = (df.iloc[:, 1] == 'M').values

X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=0.30, random_state=70, shuffle=True)

rf = RandomForest(forest_size=5,max_depth=2)
rf.train(X_train, y_train)
print("My Accuracy:{0:.2f}%".format(rf.test(X_test, y_test)))

# use sklearn


clf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=0)
clf.fit(X_train, y_train)
print("Sklearn Accuracy:{0:.2f}%".format(clf.score(X_test, y_test)*100))
