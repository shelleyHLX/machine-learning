# coding: utf-8
# Author: shelley
# 2020/5/1214:48

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random


def Gradient_Ascent_test():
    """
    梯度上升算法测试函数
        求函数f(x) = -x^2+4x的极大值
    :return:
    """
    # f(x)的值
    def f_value(x):
        return -x**2+4*x
    # f(x)的导数
    def f_derivative(x_old):
        return -2 * x_old + 4

    # 初始值，给一个小于x_new的值
    x_old = -109
    # 梯度上升算法初始值，即从(0, 0)开始
    x_new = 4
    # 步长，也就是学习速率，控制更新的幅度
    alpha = 0.01
    # 精度，也就是更新阈值
    presision = 0.00000001
    while abs(x_new - x_old) > presision:
        x_old = x_new
        # 更新下标的值,x:=x+alpha*f'(x)
        x_new = x_old + alpha * f_derivative(x_old)
    # 打印最终求解的极值近似值
    print(x_new, f_value(x_new))


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


def plotBestFit(weights):
    """
    绘制数据集
    :param weights:  权重参数数组
    :return: None
    """
    # 加载数据集
    dataMat, labelMat = loadDataSet()
    # 转换成numpy的array数组
    dataArr = np.array(dataMat)
    # 数据个数
    # 例如建立一个4*2的矩阵c，c.shape[1]为第一维的长度2， c.shape[0]为第二维的长度4
    n = np.shape(dataMat)[0]
    # 正样本
    xcord1 = []
    ycord1 = []
    # 负样本
    xcord2 = []
    ycord2 = []
    # 根据数据集标签进行分类
    for i in range(n):
        if int(labelMat[i]) == 1:
            # 1为正样本
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            # 0为负样本
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    # 新建图框
    fig = plt.figure()
    # 添加subplot
    ax = fig.add_subplot(111)
    # 绘制正样本
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    # 绘制负样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)
    # x轴坐标
    x = np.arange(-3.0, 3.0, 0.1)
    # w0*x0 + w1*x1 * w2*x2 = 0
    # x0 = 1, x1 = x, x2 = y
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    # 绘制title
    plt.title('BestFit')
    # 绘制label
    plt.xlabel('x1')
    plt.ylabel('y2')
    # 显示
    plt.show()


def sigmoid(inX):
    """
    sigmoid函数
    :param inX: 数据
    :return:
    """
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMath, classLabels):
    """
    梯度上升法
    :param dataMath: 数据集,(n_samples, 3)
    :param classLabels: 数据标签(n_samples,)
    :return:
    weights.getA() - 求得的权重数组（最优参数）(3,)
    weights_array - 每次更新的回归系数 (maxCycles,3)
    """
    # 转换成numpy的mat(矩阵)
    dataMatrix = np.mat(dataMath)
    # 转换成numpy的mat(矩阵)并进行转置
    labelMat = np.mat(classLabels).transpose()
    # 返回dataMatrix的大小，m为行数，n为列数
    m, n = np.shape(dataMatrix)
    # 移动步长，也就是学习效率，控制更新的幅度
    alpha = 0.01
    # 最大迭代次数
    maxCycles = 500
    weights = np.ones((n, 1))  # (3,1)
    weights_array = np.array([])
    for k in range(maxCycles):
        # 梯度上升矢量化公式,特征*权重，得到sigmoid的输入，sigmoid的输出即是标签
        h = sigmoid(dataMatrix * weights)  # (n_samples,3)*(3,1)=(n_samples,1)
        error = labelMat - h  # 预测标签和真实标签的误差
        # f(x)=y-sigmoid(x*w),f'(x)=s(x*w)*(1-s(x*w))*w
        weights = weights + alpha * dataMatrix.transpose() * error
        # numpy.append(arr, values, axis=None):
        # 就是arr和values会重新组合成一个新的数组，做为返回值。
        # 当axis无定义时，是横向加成，返回总是为一维数组
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxCycles, n)
    # 将矩阵转换为数组，返回权重数组
    # mat.getA()将自身矩阵变量转化为ndarray类型变量
    return weights.getA(), weights_array


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    改进的随机梯度上升法
    :param dataMatrix: 数据数组(n_samples,3)
    :param classLabels: 数据标签(n_samples,)
    :param numIter: 迭代次数
    :return:
    weights - 求得的回归系数数组（最优参数）
    weights_array - 每次更新的回归系数
    """
    # 返回dataMatrix的大小，m为行数，n为列数
    m, n = np.shape(dataMatrix)
    # 参数初始化
    weights = np.ones(n)
    weights_array = np.array([])
    for j in range(numIter):
        dataIndex = list(range(m))  # 样本个数
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
            # 添加返回系数到数组中当axis为0时，数组是加在下面（列数要相同）
            weights_array = np.append(weights_array, weights, axis=0)
            # 删除已使用的样本
            del (dataIndex[randIndex])
    # 改变维度
    weights_array = weights_array.reshape(numIter * m, n)
    # 返回
    return weights, weights_array


def plotWeights(weights_array1, weights_array2):
    """
    绘制回归系数与迭代次数的关系
    :param weights_array1: 回归系数数组1
    :param weights_array2: 回归系数数组2
    :return:
    """
    # 设置汉字格式为14号简体字
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=20)
    # 将fig画布分隔成1行1列，不共享x轴和y轴，fig画布的大小为（20, 10）
    # 当nrows=3，ncols=2时，代表fig画布被分为6个区域，axs[0][0]代表第一行第一个区域
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False,
                            figsize=(15, 7))
    # x1坐标轴的范围
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'改进的梯度上升算法，回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'w0', FontProperties=font)
    plt.setp(axs0_title_text, size=10, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=10, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'w1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=10, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_title_text = axs[2][0].set_title(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'w2', FontProperties=font)
    plt.setp(axs2_title_text, size=10, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=10, weight='bold', color='black')

    # x2坐标轴的范围
    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'梯度上升算法，回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'w0', FontProperties=font)
    plt.setp(axs0_title_text, size=10, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=10, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'w1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=10, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_title_text = axs[2][1].set_title(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'w2', FontProperties=font)
    plt.setp(axs2_title_text, size=10, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=10, weight='bold', color='black')

    plt.show()


if __name__ == '__main__':
    # 测试简单梯度上升法
    # Gradient_Ascent_test()
    # exit(0)
    # 加载数据集,(n_samples, 3), (n_sampels, )
    dataMat, labelMat = loadDataSet()
    # 训练权重
    weights2, weights_array2 = gradAscent(dataMat, labelMat)
    print(weights2)
    # print(weights_array2)
    # 新方法训练权重
    weights1, weights_array1 = stocGradAscent1(np.array(dataMat), labelMat)
    print(weights1)
    # 绘制数据集中的y和x的散点图
    # print(gradAscent(dataMat, labelMat))
    plotWeights(weights_array1, weights_array2)
