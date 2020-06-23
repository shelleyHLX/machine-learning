# coding: utf-8
# Author: shelley
# 2020/5/1210:34

# 1）sklearn实现二分类
from sklearn import tree

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print(clf.predict([[2, 2]]))  # 预测属于哪个类
print(clf.predict_proba([[2, 2]]))  # 预测属于每个类的概率

# 2）sklearn实现多分类
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# 3)sklearn实现树回归
# 注意：回归的时候y的值应该是浮点数，而不是整数值
from sklearn import tree

X = [[0, 0], [2, 2]]
y = [0.5, 0.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
print(clf.predict([[1, 1]]))