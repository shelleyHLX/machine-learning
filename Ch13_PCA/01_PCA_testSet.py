# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename, delim='\t'):
    """
    解析文本数据
    :param filename: 文件名
    :param delim: 每一行不同特征数据之间的分隔方式，默认是tab键‘\t’
    :return: j将float型数据值列表转化为矩阵返回
    """
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return np.mat(datArr)


def pca(dataMat, topNfeat=4096):
    """
    PCA特征维度压缩函数
    :param dataMat: 数据集数据
    :param topNfeat: 需要保留的特征维度，即要压缩成的维度数，默认4096
    :return:
    lowDDataMat - 压缩后的数据矩阵
    reconMat - 压缩后的数据矩阵反构出原始数据矩阵
    """
    # 求矩阵每一列的均值
    meanVals = np.mean(dataMat, axis=0)
    # 数据矩阵每一列特征减去该列特征均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵，处以n-1是为了得到协方差的无偏估计
    # cov(x, 0) = cov(x)除数是n-1(n为样本个数)
    # cov(x, 1)除数是n
    covMat = np.cov(meanRemoved, rowvar=0)
    # 计算协方差矩阵的特征值及对应的特征向量
    # 均保存在相应的矩阵中
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # sort():对特征值矩阵排序(由小到大)
    # argsort():对特征矩阵进行由小到大排序，返回对应排序后的索引
    eigValInd = np.argsort(eigVals)
    # 从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的索引
    eigValInd = eigValInd[: -(topNfeat+1): -1]
    # 将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
    redEigVects = eigVects[:, eigValInd]
    # 将去除均值后的矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    lowDDataMat = meanRemoved * redEigVects
    # 利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
    # 此处用转置和逆的结果一样redEigVects.I
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    print(reconMat)
    # 返回压缩后的数据矩阵及该矩阵反构出原始数据矩阵
    return lowDDataMat, reconMat


if __name__ == '__main__':
    dataMat = loadDataSet('testSet.txt')
    print(np.shape(dataMat))  # (1000,2)
    lowDmat, reconMat = pca(dataMat, 1)
    print(np.shape(lowDmat))  # (1000,1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=90, c='red')
    plt.show()


