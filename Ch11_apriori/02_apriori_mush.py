# coding: utf-8
# Author: shelley
# 2020/5/219:47


def createC1(dataSet):
    """
    创建数据集中所有单一元素组成的集合
    :param dataSet:
    :return:，C1是一个集合的集合，如 {{0}，{1}，{2}，_..}，每次添加的都是单个项构成的集合{0}、{1}、{2}
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # frozenset() 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素。
    return list(map(frozenset, C1))


def scanD(D, Ck, minSupport):
    """
    从C1生成L1
    :param D: 原始数据集
    :param Ck: 上一步生成的单元素数据集
    :param minSupport: 最小支持度
    :return:
    retList - 符合条件的元素
    supportData - 符合条件的元素及其支持率组成的字典
    """
    ssCnt = {}  # 在样本中的单元素
    # 以下这一部分统计每一个单元素出现的次数
    for tid in D:  # 遍历全体样本中的每一个元素
        for can in Ck:  # 遍历单元素列表中的每一个元素，判断每个元素是否在样本元素的子集中
            # s.issubset( x ) 判断集合s是否是集合x子集
            #  找出所有样本集中包含单元素的样本
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    # 获取样本中的元素个数
    numItems = float(len(D))
    retList = []
    supportData = {}
    # 遍历每一个单元素
    for key in ssCnt:
        # 计算每一个单元素的支持率
        support = ssCnt[key] / numItems
        # 若支持率大于最小支持率
        if support >= minSupport:
            # insert() 函数用于将指定对象插入列表的指定位置。
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    """
    组合向上合并,
    前k-2项相同时，将两个集合合并
    :param Lk: 频繁项集列表[]
    :param k: 项集元素个数
    :return: 符合条件的元素
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        # 两两组合遍历
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            # 若两个组合的前k-2个项相同时，则将这两个集合合并
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport=0.5):
    """
    apriori算法
    :param dataSet: 原始数据集
    :param minSupport: 最小支持度
    :return:
    L - 符合条件的元素
    supportData - 符合条件的元素及其支持率组成的字典
    """
    # 创建数据集中所有单一元素组成的集合保存在C1中
    C1 = createC1(dataSet)
    # 将数据集元素转为set集合然后将结果保存为列表
    D = list(map(set, dataSet))
    # 从C1生成L1并返回符合条件的元素，符合条件的元素及其支持率组成的字典
    # 样本中满足最小支持度的样本
    # k=1，[单元素,],{单元素:最小支持度}
    L1, supportData = scanD(D, C1, 0.5)
    # 将符合条件的元素转换为列表保存在L中L会包含L1、L2、L3......
    L = [L1]
    k = 2  # k和L的下标相差1
    # L[n]就代表n+1元素集合,例如L[0]代表1个元素的集合
    # L[0]=[frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})]
    while(len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)  # k=2使用L[0]的数据,Ck的集合
        Lk, supK = scanD(D, Ck, minSupport)
        # dict.update(dict2) 字典update()函数把字典dict2的键/值对更新到dict里。
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
L, suppData = apriori(mushDatSet, minSupport=0.3)

for item in L[1]:
    if item.intersection('2'):
        print(item)



