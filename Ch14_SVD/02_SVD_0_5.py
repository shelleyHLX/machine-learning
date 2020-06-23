# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg


def imgLoadData(filename):
    """
    加载并转换数据
    :param filename: 文件名
    :return: 图像矩阵
    """
    myl = []
    # 打开文本文件，并从文件以数组方式读入字符
    for line in open(filename).readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    return myMat


def analyse_data(Sigma, loopNum=20):
    """
    分析Sigma的长度取值
    :param Sigma: Sigma值
    :param loopNum: 循环次数
    :return: 总方差的集合（总能量值）
    """
    # 总方差的集合（总能量值）
    Sig2 = Sigma ** 2
    SigmaSum = np.sum(Sig2)
    for i in range(loopNum):
        # 根据自己的业务情况，进行处理，设置对应的Sigma次数
        # 通常保留矩阵80%~90%的能量，就可以得到重要的特征并去除噪声
        SigmaI = np.sum(Sig2[:i+1])
        print('主成分：%s, 方差占比: %s%%' % (format(i+1, '2.0f'), format(SigmaI / SigmaSum * 100, '4.2f')))


def loadExData():
    """
    加载数据,菜肴矩阵
    行：代表人
    列：代表菜肴名词
    值：代表人对菜肴的评分，0代表未评分
    :return: 菜肴矩阵
    """
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def standEst(dataMat, user, simMeas, item):
    """
    基于物品相似度的推荐引擎
    计算某用户未评分物品，以对该物品和其他物品评分的用户的物品相似度，
    然后进行综合评分
    :param dataMat: 训练数据集
    :param user: 用户编号
    :param simMeas: 相似度计算方法
    :param item: 未评分的物品编号
    :return: 评分（0~5之间的值）
    """
    # 得到数据集中的物品数目
    n = np.shape(dataMat)[1]
    # 初始化两个评分值
    simTotal = 0.0
    ratSimTotal = 0.0
    # 遍历行中的每个物品（对用户评过分的物品进行遍历，并将它与其他物品进行比较）
    for j in range(n):
        userRating = dataMat[user, j]
        # 如果某个物品的评分值为0，则跳过这个物品
        if userRating == 0:
            continue
        # 寻找两个用户都评级的物品
        # 变量overLap给出的是两个物品当中已经被评分的那个元素的索引ID
        # logical_and计算x1和x2元素的逻辑与
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        # 如果相似度为0，则两者没有任何重合元素，终止本次循环
        if len(overLap) == 0:
            similarity = 0
        # 如果存在重合的物品，则基于这些重合物重新计算相似度
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        # 相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积
        # similarity 用户相似度   userRating 用户评分
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    # 通过除以所有的评分和，对上述相似度评分的乘积进行归一化，使得最后评分在0~5之间，这些评分用来对预测值进行排序
    else:
        return ratSimTotal / simTotal

def svdEst(dataMat, user, simMeas, item):
    """
    基于SVD的评分估计
    :param dataMat: 训练数据集
    :param user: 用户编号
    :param simMeas: 相似度计算方法
    :param item: 未评分的物品编号
    :return: 评分（0~5之间的值）
    """
    # 得到数据集中的物品数目
    n = np.shape(dataMat)[1]
    # 初始化两个评分值
    simTotal = 0.0
    ratSimTotal = 0.0
    # 奇异值分解
    # 在SVD分解之后，我们只利用包含90%能量值的奇异值，这些奇异值会以Numpy数组形式得以保存
    U, Sigma, VT = linalg.svd(dataMat)
    # 如果要进行矩阵运算，就必须要用这些奇异值构造出一个对角阵
    Sig4 = np.mat(np.eye(4) * Sigma[: 4])
    # 利用U矩阵将物品转换到低维空间中，构建转换后的物品（物品的4个主要特征）
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    # 遍历行中的每个物品（对用户评过分的物品进行遍历，并将它与其他物品进行比较）
    for j in range(n):
        userRating = dataMat[user, j]
        # 如果某个物品的评分值为0，则跳过这个物品
        if userRating == 0:
            continue
        # 相似度的计算也会作为一个参数传递给该函数
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        # 相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积
        # similarity 用户相似度   userRating 用户评分
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    # 通过除以所有的评分和，对上述相似度评分的乘积进行归一化，使得最后评分在0~5之间，这些评分用来对预测值进行排序
    else:
        return ratSimTotal / simTotal

def ecludSim(inA, inB):
    """
    欧氏距离
    :param inA: A数据向量
    :param inB: B数据向量
    :return: 相似度
    """
    return 1.0 / (1.0 + linalg.norm(inA - inB))


def pearsSim(inA, inB):
    """
    皮尔逊相关系数
    :param inA: A数据向量
    :param inB: B数据向量
    :return: 相似度
    """
    # 此时两个向量完全相关
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    """
    余弦相似度
    :param inA: A数据向量
    :param inB: B数据向量
    :return: 相似度
    """
    num = float(inA.T * inB)
    denom = linalg.norm(inA) * linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    """
    推荐引擎
    :param dataMat: 训练数据集
    :param user: 用户编号
    :param N: 产生N个推荐结果
    :param simMeas: 相似度计算方法
    :param estMethod: 推荐引擎方法
    :return: 评分（0~5之间的值）
    """
    # 寻找未评级的物品
    # 对给定的用户建立一个未评分的物品列表
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    # 如果不存在未评分物品，那么就退出函数
    if len(unratedItems) == 0:
        return ('you rated everything')
    # 物品的编号和评分值
    itemScores = []
    # 在未评分的物品上进行循环
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        # 寻找前N个未评级的物品，调用standEst()来产生该物品的预测得分，该物品的编号和估计值会放在一个元素列表itemScores中
        itemScores.append((item, estimatedScore))
    # 返回元素列表，第一个就是最大值
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[: N]

def printMat(inMat, thresh=0.8):
    """
    打印矩阵
    :param inMat: 图像矩阵
    :param thresh:
    :return:
    """
    # 由于矩阵保护了浮点数，因此定义浅色和深色，遍历所有矩阵元素，大于阈值打印1，否则打印0
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1,end='')
            else:
                print(0,end='')
        print('')

def imgCompress(numSV=3, thresh=0.8):
    """
    实现图像压缩，允许基于任意给定的奇异值数目来重构图像
    :param numSV: Sigma长度
    :param thresh:判断的阈值
    :return:
    """
    myMat = imgLoadData('0_5.txt')
    print('****original matrix****')
    printMat(myMat, thresh)
    # 对原始图像进行SVD分解并重构图像
    U, Sigma, VT = linalg.svd(myMat)
    # 分析插入的Sigma长度
    analyse_data(Sigma, 20)
    SigRecon = np.mat(np.eye(numSV) * Sigma[: numSV])
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print('****reconstructed matrix using %d singular values ****' % numSV)
    printMat(reconMat, thresh)
    
if __name__ == '__main__':
    imgCompress(2)
    
