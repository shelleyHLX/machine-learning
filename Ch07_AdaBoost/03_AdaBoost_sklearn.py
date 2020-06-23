# coding: utf-8
# Author: shelley
# 2020/5/1311:43
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def loadDataSet(fileName):
    """
    加载文件
    :param fileName: 文件名
    :return:
    dataMat - 数据矩阵
    labelMat - 数据标签
    """
    # 特征个数
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


from sklearn.svm import SVC


if __name__ == '__main__':
    dataArr, classLabels = loadDataSet('horseColicTraining2.txt')
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    # bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=10)
    bdt = AdaBoostClassifier(SVC(C=200.0, kernel='rbf'), algorithm="SAMME", n_estimators=10)
    bdt.fit(dataArr, classLabels)
    predictions = bdt.predict(dataArr)
    errArr = np.mat(np.zeros((len(dataArr), 1)))
    for i in range(len(classLabels)):
        if classLabels[i] != predictions[i]:
            errArr[i] = 1
    print('训练集的错误率:%.3f%%' % float(errArr.sum() / len(dataArr) * 100))
    predictions = bdt.predict(testArr)
    errArr = np.mat(np.zeros((len(testArr), 1)))
    for i in range(len(testLabelArr)):
        if testLabelArr[i] != predictions[i]:
            errArr[i] = 1
    print('测试集的错误率:%.3f%%' % float(errArr.sum() / len(testArr) * 100))

# 训练集的错误率:0.334%
# 测试集的错误率:28.358%

