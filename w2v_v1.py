# -*- coding: utf-8 -*-
"""
    word2vec -> sentence2vec -> cos-similarity
"""
import re
import math
import jieba
import string
import numpy as np
import pandas as pd
from zhon.hanzi import punctuation
from gensim.models import Word2Vec, KeyedVectors

# jieba添加用户词典
user_words_file = 'user_words.txt'
user_words = open(user_words_file, 'r', encoding='utf8').readlines()
user_words = [uw.strip() for uw in user_words]
for uw in user_words:
    jieba.add_word(uw)

def get_train_datas():
    '''
        得到训练文本数据集,sku_names是所有训练文本数据集
    :return:
    '''
    datas = pd.read_csv("data/mobile.csv")
    sku_names = datas["sku_name"]
    return sku_names

def get_test_datas():
    '''
        得到测试文本数据集，keywords是所有测试文本数据集
    :return:
    '''
    datas = pd.read_excel("data/keywords.xls")
    keywords = datas["关键词"]
    return keywords

def get_text_jieba(texts):
    '''
        得到jieba分词结果
    :param train_texts:
    :return:
    '''
    #stop_words_dict = get_stop_words()

    all_doc_list = []
    for doc in texts:
        if type(doc) == float:
            # 针对sku_name数据缺失值nan问题
            doc = "手机错误"

        # 去标点符号
        doc = re.sub("[%s%s]+" % (punctuation, string.punctuation), "", doc)
        # 统一字符大写
        doc = doc.upper()

        if doc == '':
            doc = "手机错误"

        doc_list = []
        for word in jieba.cut(doc.strip()):
            if len(word) > 0 and word != ' ':
                doc_list.append(word)
        all_doc_list.append(doc_list)

    return all_doc_list

def sent2vec(sentence, model):
    """
        得到句子向量
    :param sentence:
    :param model:
    :return:
    """
    M = []
    for word in sentence:
        try:
            M.append(model[word])
        except:
            # OOV，设为0
            ignore = [0] * 100 # 100 为词向量维度
            M.append(ignore)
    M = np.array(M)
    V = M.sum(axis=0)
    return V / np.sqrt((V ** 2).sum())

def cos_sim(sentence1, sentence2):
    """
        计算两个句子的余弦相似度
    :param sentence1:
    :param sentence2:
    :return:
    """
    if len(sentence1) != len(sentence2):
        return None
    up = 0.0
    down_a = 0.0
    down_b = 0.0
    for s1, s2 in zip(sentence1, sentence2):
        up += s1*s2
        down_a += s1 ** 2
        down_b += s2 ** 2
    down = math.sqrt(down_a*down_b)
    if down == 0.0:
        return -1
    else:
        return up / down

def test():
    model = KeyedVectors.load_word2vec_format('models/skuname100.vector', binary=False)

    # 构建匹配语料库 398872 samples
    sku_names_texts = get_train_datas()
    sku_names_jieba = get_text_jieba(sku_names_texts)
    sku_names_vectors = []
    for sku_names in sku_names_jieba:
        sku_name_sent2vec = sent2vec(sku_names, model)
        sku_names_vectors.append(sku_name_sent2vec)
    print(len(sku_names_texts), len(sku_names_jieba))
    print(sku_names_jieba[0])
    print(len(sku_names_vectors))

    # 测试数据 1000 samples
    keywords_texts = get_test_datas()
    keywords_jieba = get_text_jieba(keywords_texts)
    keywords_vectors = []
    for keywords in keywords_jieba:
        keywords_sent2vec = sent2vec(keywords, model)
        keywords_vectors.append(keywords_sent2vec)
    print(len(keywords_texts))
    print(len(keywords_vectors))

    # 计算相似度
    for i, keywords_vec in enumerate(keywords_vectors):
        sim_scores = []
        for sku_names_vec in sku_names_vectors:
            sim = cos_sim(keywords_vec, sku_names_vec)
            sim_scores.append(sim)
        idx = sim_scores.index(max(sim_scores))
        print(i, "||", keywords_texts[i], "||", sku_names_texts[idx])

        with open("result/w2v_v1_results.txt", 'a', encoding='utf8') as wf:
            wf.write(str(i) + "||" + keywords_texts[i] + "||" + sku_names_texts[idx] + "\n")

if __name__ == '__main__':
    test()