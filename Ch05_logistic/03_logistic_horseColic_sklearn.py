# coding: utf-8
# Author: shelley
# 2020/5/1214:56

from sklearn.linear_model import LogisticRegression

def colicSklearn():
    """
    使用Sklearn构建Logistic回归分类器
    :return:
    """
    # 打开训练集
    frTrain = open('horseColicTraining.txt')
    # 打开测试集
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    # fit(X,y) Fit the model according to the given training data
    classifier = LogisticRegression(solver='liblinear', max_iter=10).fit(trainingSet, trainingLabels)
    # score(X,y) Returns the mean accuracy on the given test data and labels
    test_accurcy = classifier.score(testSet, testLabels) * 100
    print("正确率为：%f%%" % test_accurcy)  # 73.134328%


if __name__ == '__main__':
    colicSklearn()