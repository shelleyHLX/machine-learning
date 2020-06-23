# -*- coding: utf-8 -*-


from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np


def loadDataSet(filename):
    """
    加载数据
    :param filename: 文件名
    :return:
    xArr - x数据集
    yArr - y数据集
    """
    # 计算特征个数，由于最后一列为y值所以减一
    numFeat = len(open(filename).readline().split('\t')) - 1
    xArr = []
    yArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr


def lwlr(testPoint, xArr, yArr, k=1.0):
    """
    使用局部加权线性回归计算回归系数w
    :param testPoint: 测试样本点
    :param xArr: x数据集
    :param yArr: y数据集
    :param k: 高斯核的k，自定义参数
    :return: 回归系数
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]  # 样本个数
    # 创建加权对角阵
    weights = np.mat(np.eye((m)))
    # 求测试样例中与所有数据集中的权重
    for j in range(m):  # 遍历每个样本
        # 高斯核
        diffMat = testPoint - xMat[j, :]  # 其中一个样本-其中一个样本点
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)  # weights是每个样本的权重(m,m)*(m,feas)
    # 求矩阵的行列式
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵，不能求逆")
        return
    # .I求逆矩阵
    ws = (xTx.I) * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    """
    局部加权线性回归测试
    :param testArr: 测试数据集
    :param xArr: x数据集
    :param yArr: y数据集
    :param k: 高斯核的k,自定义参数
    :return: 回归系数
    """
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def plotlwlrRegression():
    """
    绘制多条局部加权回归曲线
    :return: None
    """
    # 设置中文字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)
    xArr, yArr = loadDataSet('ex0.txt')
    yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)
    yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    # 排序，返回索引值
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(10, 8))
    axs[0].plot(xSort[:, 1], yHat_1[srtInd], c='red')
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c='red')
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c='red')
    axs[0].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    axs[1].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    axs[2].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0', FontProperties=font)
    axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01', FontProperties=font)
    axs2_title_text = axs[2].set_title(u'局部加权回归曲线,k=0.003', FontProperties=font)
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()


if __name__ == '__main__':
    plotlwlrRegression()

