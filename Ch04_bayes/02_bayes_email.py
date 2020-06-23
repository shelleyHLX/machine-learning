# coding: utf-8
# Author: shelley
# 2020/5/1213:53
import numpy as np


def getTopWords(ny,sf):
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print ("SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print (item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print ("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print (item[0])

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

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


def textParse(bigString):
    """
    对字符串进行处理
    :param bigString:字符串
    :return:string list ['we','are',...]
    """
    import re
    # 用特殊符号作为切分标志进行字符串切分，即非字母、非数字
    # \W* 0个或多个非字母数字或下划线字符（等价于[^a-zA-Z0-9_]）
    listOfTokens = re.split(r'\W*', bigString)
    # 除了单个字母，例如大写I，其他单词变成小写，去掉少于两个字符的字符串
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def bagOfWords2VecMN(vocabList, inputSet):
    """

    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
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
    p0Denom = 2.0  # 类别0的所有样本的单词总数
    p1Denom = 2.0
    for i in range(numTrainDocs):  # 每个样本
        if trainCategory[i] == 1:  # 类别1
            p1Num += trainMatrix[i]  # 每个单词的个数，[0,0,1,...]+[0,1,1,...]
            p1Denom += sum(trainMatrix[i])
        else:  # 类别0
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)  # 类别1中一个单词出现的次数
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """

    :param vec2Classify:(vocal,1)
    :param p0Vec:非侮辱类的单词出现概率(vocal,1)
    :param p1Vec:侮辱类的单词出现概率(vocal,1)
    :param pClass1:文档属于侮辱类的概率int
    :return:int
    """
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        # 读取每个垃圾邮件的所有内容，并以字符串转换成字符串列表
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)  # [['we', 'do',...],[]]
        fullText.extend(wordList)  # ['we', 'do',...,'me',...]
        classList.append(1)
        try:
            wordList = textParse(open('email/ham/%d.txt' % i).read())
        except Exception as ex:
            print(i)
            exit(0)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # 计算所有词汇
    trainingSet = list(range(50)); testSet=[]
    # 从50个邮件中，随机挑选出40个作为训练集，10个作为测试集
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  # 在训练集列表中删除添加到测试集的索引值

    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:
        # 将生成的词集模型添加到训练集矩阵中
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # 训练朴素贝叶斯模型
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    # 测试样本
    errorCount = 0
    for docIndex in testSet:
        # 生成的词集模型
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print ("分类错误的测试样本：",docList[docIndex])
    print("错误率：%.2f%%" % (float(errorCount) / len(testSet) * 100))


if __name__ == '__main__':
    spamTest()
    # i = 2
    # text = open('email/spam/%d.txt' % i).read()
    # print(str(text))