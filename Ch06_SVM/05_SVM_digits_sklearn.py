# coding: utf-8
# Author: shelley
# 2020/5/1311:37
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:02:27 2018

@author: wzy
"""
import numpy as np
import operator
from os import listdir
from sklearn.svm import SVC

"""
函数说明：将32*32的二进制图像转换为1*1024向量

Parameters:
    filename - 文件名

Returns:
    returnVect - 返回二进制图像的1*1024向量

Modify:
    2018-07-25
"""


def img2vector(filename):
    # 创建1*1024零向量
    returnVect = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(32):
        # 读取一行数据
        lineStr = fr.readline()
        # 每一行的前32个数据依次存储到returnVect中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    # 返回转换后的1*1024向量
    return returnVect


"""
函数说明：手写数字分类测试

Parameters:
    None

Returns:
    None

Modify:
    2018-07-25
"""


def handwritingClassTest():
    # 测试集的Labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir('trainingDigits')
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵（全零阵），测试集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别填加到hwLabels中
        hwLabels.append(classNumber)
        # 将每一个文件的1*1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % (fileNameStr))
    # 构造SVM
    clf = SVC(C=200, kernel='rbf')
    # SVC()
    # fit(X, y):Fit SVM model according to given training data
    clf.fit(trainingMat, hwLabels)
    # 返回testDigits目录下的文件列表
    testFileList = listdir('testDigits')
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将每一个文件的1*1024数据存储到vectorUndertest矩阵中
        vectorUndertest = img2vector('testDigits/%s' % (fileNameStr))
        # 获得预测结果
        # predict(X):Perform classification on samples in X
        classifierResult = clf.predict(vectorUndertest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))


if __name__ == '__main__':
    handwritingClassTest()
