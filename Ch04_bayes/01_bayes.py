'''
Created on Oct 19, 2010

@author: Peter
'''
import numpy as np

def loadDataSet():
    """
    创建实验样本
    :return:
    postingList - 实验样本切分的词条
    classVec - 类别标签向量
    """
    # 切分的词条
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0,1,0,1,0,1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec
                 

def setOfWords2Vec(vocabList, inputSet):
    """
    根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
    :param vocabList: createVocabList返回的列表
    :param inputSet: 切分的词条列表
    :return: 文档向量，词袋模型[0,1,0,0,...]
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print ("词 %s 不在词汇表中" % word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    """
    朴素贝叶斯分类器训练函数
    :param trainMatrix:文档矩阵，（samples,vocabulary）[[0,0,1,...],[]]
    :param trainCategory:训练类标签向量，[0,0,1]，1表示侮辱类
    :return:
    p1Vect - 侮辱类的单词出现概率(vocal,1)
    p0Vect - 非侮辱类的单词出现概率(vocal,1)
    pAbusive - 文档属于侮辱类的概率int
    """
    numTrainDocs = len(trainMatrix)  # 训练样本的个数
    numWords = len(trainMatrix[0])  # 词汇个数
    pAbusive = sum(trainCategory)/float(numTrainDocs)  # 文档属于侮辱类的概率
    p0Num = np.ones(numWords)  # (1,vocal)
    p1Num = np.ones(numWords)  # (1,vocal)
    # 分母初始化为2，拉普拉斯平滑
    p0Denom = 2.0  # 类别0的所有样本的每个单词的出现的总数
    p1Denom = 2.0
    for i in range(numTrainDocs):  # 每个样本
        if trainCategory[i] == 1:  # 类别1
            p1Num += trainMatrix[i]  # 每个单词的个数，[0,0,1,...]+[0,1,1,...]
            p1Denom += sum(trainMatrix[i])
        else:  # 类别0
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)  # 类别1中一个单词出现的次数/类别1中所有单词出现的个数
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """

    :param vec2Classify:(vocal,1)[0,1,0,1]
    :param p0Vec:非侮辱类的单词出现概率(vocal,1)
    :param p1Vec:侮辱类的单词出现概率(vocal,1)
    :param pClass1:文档属于侮辱类的概率int
    :return:int
    """
    # 如果出现该词，则将该词的概率加上
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0


def createVocabList(dataSet):
    """
    将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
    :param dataSet:整理的样本数据集
    :return:返回不重复的词条列表，也就是词汇表
    """
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 两个set的union
    return list(vocabSet)

def estingNB():
    """

    :return:
    """
    listOPosts,listClasses = loadDataSet()  # 数据，类别

    myVocabList = createVocabList(listOPosts)  # 创建词汇表
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # (vocal,1),(vocal,1),int
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))


if __name__ == '__main__':
    estingNB()


