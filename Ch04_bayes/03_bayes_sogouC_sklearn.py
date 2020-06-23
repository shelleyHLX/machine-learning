# coding: utf-8
# Author: shelley
# 2020/5/1214:16
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import os
import random
import jieba


def TextProcessing(folder_path, test_size=0.2):
    """
    中文文本处理
        将文件夹内部的所有txt文档分词并存储在data_list中，
        将txt上一级文件夹名称存储在class_list中
    :param folder_path: 文本存放的路径
    :param test_size: 测试集占比，默认占所有数据集的20%
    :return:
    all_words_list - 按词频降序排序的训练集列表
    train_data_list - 训练集列表
    test_data_list - 测试集列表
    train_class_list - 训练集标签列表
    test_class_list - 测试集标签列表
    """
    # 查看folder_path下的文件
    # os.listdir(path)方法用于返回指定的文件夹包含的文件或文件夹的名字列表。这个列表以字母顺序。
    # 它不包括'.'和'..'即使它在文件夹中。
    folder_list = os.listdir(folder_path)
    # 数据集数据
    data_list = []
    # 数据集类别
    class_list = []
    # 遍历每个子文件夹
    for folder in folder_list:
        # 根据子文件夹，生成新的路径
        # os.path.join路径名拼接即folder_path+folder从而生成新的路径，
        # 可以遍历每一个文件
        new_folder_path = os.path.join(folder_path, folder)
        # 存放子文件夹下的txt文件列表
        files = os.listdir(new_folder_path)
        j = 1
        # 遍历每个txt文件
        for file in files:
            # 每类txt样本数最多100个
            if j > 100:
                break
            # 打开txt文件
            with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as f:
                # 读取txt文件内容
                raw = f.read()
            # 精简模式，返回一个可迭代的generator
            # jieba.cut方法接受两个输入参数：1）第一个参数为需要分词的字符串
            # 2）cut_all参数用来控制是否采用全模式
            word_cut = jieba.cut(raw, cut_all=False)
            # generator转换为list
            word_list = list(word_cut)
            # 存储经过分割以后的词语列表
            data_list.append(word_list)
            # 存储上一级文件夹名称
            class_list.append(folder)
            # 自增
            j += 1
    # zip压缩合并，将数据与标签对应压缩
    # zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，
    # 然后返回由这些元组组成的列表,如果各个迭代器的元素个数不一致，
    # 则返回列表长度与最短对象相同，利用*号操作符，可以将元组解压为列表
    # python3中zip()返回一个对象，如需展示列表，需手动list()转换
    data_class_list = list(zip(data_list, class_list))
    # 将data_class_list乱序，shuffle()方法将序列或元组所有元素随机排序
    random.shuffle(data_class_list)
    # 训练集和测试集切分的索引值
    index = int(len(data_class_list) * test_size) + 1
    # 训练集
    train_list = data_class_list[index:]
    # 测试集
    test_list = data_class_list[:index]
    # print('zip: ', train_list[0])
    # 训练集解压缩为列表
    train_data_list, train_class_list = zip(*train_list)
    # 测试集解压缩为列表
    # print('unzip: ', train_data_list[0], train_class_list[0])
    test_data_list, test_class_list = zip(*test_list)
    # 统计训练集词频
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                # 拉普拉斯平滑
                all_words_dict[word] = 1
    # 根据键的值倒序排序,返回元组对，[('c', 99), ('a', 44), ('b', 22)]
    # 频率从大到小排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
    # 字典解压缩为列表，['we','be',..],[22,11,...]
    all_words_list, all_words_nums = zip(*all_words_tuple_list)
    # 转换成列表
    all_words_list = list(all_words_list)
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def read_stopwords(words_file):
    """
    读取文件里的内容，并去重
    :param words_file: 文件路径
    :return: 读取的内容的set集合
    """
    # set是一个无序且不重复的元素集合
    words_set = set()
    # 打开文件
    with open(words_file, 'r', encoding='utf-8') as f:
        # 一行一行读取
        for line in f.readlines():
            # 去掉每行两边的空字符
            word = line.strip()
            # 有文本，则添加到words_set中
            if len(word) > 0:
                # 集合add方法：把要传入的元素作为一个整体添加到集合中
                # 如add('python')即为‘python’
                # 集合update方法：要把传入元素拆分，作为个体传入到集合中
                # 如update('python')即为'p''y''t''h''o''n'
                words_set.add(word)
    # 返回处理结果
    return words_set


def words_dict(all_words_list, deleteN, stopwords_set=set()):
    """
    文本特征选取
    :param all_words_list: ['we','be']，频率从高到低
    :param deleteN: 删除词频最高的deleteN个词
    :param stopwords_set: 停用词
    :return: 特征集
    """
    # 特征列表
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):  # t从deleteN开始，一直加1，到len(all_words_list)
        # feature_words的维度为1000
        if n > 1000:
            break
        # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，
        # 那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words


def TextFeatures(train_data_list, test_data_list, feature_words):
    """
    根据feature_words将文本向量化
    :param train_data_list: 训练集['我们',..]
    :param test_data_list: 测试集
    :param feature_words: 特征集
    :return:
    train_feature_list - 训练集向量化列表(samples,len(feature_words))
    test_feature_list - 测试集向量化列表
    """
    # 出现在特征集中，则置1
    def text_features(text, feature_words):
        # set是一个无序且不重复的元素集合
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    # 返回结果
    return train_feature_list, test_feature_list


def TextClassifier(train_feature_list, test_feature_list,
                   train_class_list, test_class_list):
    """
    新闻分类器
    :param train_feature_list: 训练集向量化的特征文本
    :param test_feature_list: 测试集向量化的特征文本
    :param train_class_list: 训练集分类标签
    :param test_class_list: 测试集分类标签
    :return: 分类器精度
    """
    # fit(X,y) Fit Naive Bayes classifier according to X, y
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    # score(X,y) Returns the mean accuracy on the given test data and labels
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy


def main():
    # 文本预处理
    # 训练集存放地址
    folder_path = 'SogouC/Sample'
    all_words_list, train_data_list, test_data_list, \
    train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
    # print(all_words_list)
    # 生成stopwords_set
    stopwords_file = 'stopwords_cn.txt'
    stopwords_set = read_stopwords(stopwords_file)
    # 词频出现前100的删除
    # feature_words = words_dict(all_words_list, 100, stopwords_set)
    # print(feature_words)
    test_accuracy_list = []
    # 0 20 40 60 ... 980，测试去除几个频率最高的词的效果好
    deleteNs = range(0, 1000, 20)
    best_Ns = -1
    best_accu = 0.0
    for deleteN in deleteNs:
        # 去除词汇表中词的频率较高的词
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        # 获得每个样本的字符串的向量表示
        train_feature_list, test_feature_list = \
            TextFeatures(train_data_list, test_data_list, feature_words)
        test_accuracy = TextClassifier(train_feature_list, test_feature_list,
                                       train_class_list, test_class_list)
        test_accuracy_list.append(test_accuracy)
        if best_accu<test_accuracy:
            best_accu = test_accuracy
            best_Ns = deleteN
    plt.figure()
    plt.plot(deleteNs, test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accurecy')
    plt.show()
    # 删除的最好个数的测试
    feature_words = words_dict(all_words_list, best_Ns, stopwords_set)
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
    test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    test_accuracy_list.append(test_accuracy)
    ave = sum(test_accuracy_list) / len(test_accuracy_list)
    print('当删掉前%d高频词分类精度为：%.5f' % (best_Ns, ave))


if __name__ == '__main__':
    main()
