'''
Created on Oct 6, 2010

@author: Peter
'''
from numpy import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
def sigmoid(inX):
    """
    sigmoid函数
    :param inX: 数据
    :return:
    """
    return 1.0/(1+np.exp(-inX))
def loadDataSet():
    """
    加载数据
    :return:
    dataMat - 数据列表，(n_samples, 3)
    labelMat - 标签列表(n_samples,)
    """
    # 创建数据列表
    dataMat = []
    # 创建标签列表
    labelMat = []
    # 打开文件 -0.017612	14.053064	0
    #         -1.395634	4.662541	1
    fr = open('testSet.txt')
    # 逐行读取
    for line in fr.readlines():
        # 去掉每行两边的空白字符，并以空格分隔每行数据元素
        lineArr = line.strip().split()
        # 添加数据，第一个数据表示bias
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 添加标签
        labelMat.append(int(lineArr[2]))
    # 关闭文件
    fr.close()
    # 返回
    return dataMat, labelMat
def stocGradAscent0(dataMatrix, classLabels, numIter=150):
    """
    改进的随机梯度上升法
    :param dataMatrix: 数据数组
    :param classLabels: 数据标签
    :param numIter: 迭代次数
    :return: 求得的回归系数数组（最优参数）
    """
    # 返回dataMatrix的大小，m为行数，n为列数
    m, n = np.shape(dataMatrix)
    # 参数初始化
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 每次都降低alpha的大小
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机选择样本
            randIndex = int(random.uniform(0, len(dataIndex)))
            # 随机选择一个样本计算h
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            # 计算误差
            error = classLabels[randIndex] - h
            # 更新回归系数
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 删除已使用的样本
            del (dataIndex[randIndex])
    # 返回
    return weights


dataMat,labelMat=loadDataSet()
dataArr = array(dataMat)
weights = stocGradAscent0(dataArr,labelMat)

n = shape(dataArr)[0] #number of points to create
xcord1 = []; ycord1 = []
xcord2 = []; ycord2 = []

markers =[]
colors =[]
for i in range(n):
    if int(labelMat[i])== 1:
        xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
    else:
        xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(xcord,ycord, c=colors, s=markers)
type1 = ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
type2 = ax.scatter(xcord2, ycord2, s=30, c='green')
x = arange(-3.0, 3.0, 0.1)
#weights = [-2.9, 0.72, 1.29]
#weights = [-5, 1.09, 1.42]
weights = [13.03822793,   1.32877317,  -1.96702074]
weights = [4.12,   0.48,  -0.6168]
y = (-weights[0]-weights[1]*x)/weights[2]
type3 = ax.plot(x, y)
#ax.legend([type1, type2, type3], ["Did Not Like", "Liked in Small Doses", "Liked in Large Doses"], loc=2)
#ax.axis([-5000,100000,-2,25])
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()