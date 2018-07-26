# -*- coding: utf-8 -*-
"""
    TF-IDF_v1，余弦相似度，baseline
"""
import jieba
import pandas as pd
import string
import re
from zhon.hanzi import punctuation
from gensim import corpora, models, similarities

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
    all_doc_list = []
    for doc in texts:
        if type(doc) == float:
            # 针对sku_name数据缺失值nan问题
            doc = "出现错误"

        # 去标点符号
        doc = re.sub("[%s%s]+" % (punctuation, string.punctuation), "", doc)
        # 统一字符大写
        doc = doc.upper()

        doc_list = []
        for word in jieba.cut(doc.strip()):
            if len(word) > 0 and word != ' ':
                doc_list.append(word)
        all_doc_list.append(doc_list)

    return all_doc_list

def train():
    """
        句子相似度计算过程
    :return:
    """
    # 构建匹配语料库 398872 samples
    sku_names_texts = get_train_datas()
    sku_names_jieba = get_text_jieba(sku_names_texts)
    print(len(sku_names_texts), len(sku_names_jieba))
    print(sku_names_jieba[0])

    # 测试数据 1000 samples
    keywords_texts = get_test_datas()
    keywords_jieba = get_text_jieba(keywords_texts)
    print(len(keywords_texts))

    # 统计词表
    dictionary = corpora.Dictionary(sku_names_jieba)
    print(len(dictionary))

    corpus = [dictionary.doc2bow(sku_name) for sku_name in sku_names_jieba]
    tfidf = models.TfidfModel(corpus)

    # 保存与加载模型
    #tfidf.save("models/tfidf_v1")
    #tfidf = models.TfidfModel.load("models/tfidf_v1")

    # 余弦相似度
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))

    # save model

    for i, item in enumerate(keywords_jieba):
        item_vec = dictionary.doc2bow(item)
        sims = index[tfidf[item_vec]]
        idx = list(sims).index(max(list(sims)))
        print(i, "||", keywords_texts[i], "||", sku_names_texts[idx])

        with open("result/tfidf_v1_results.txt", 'a', encoding='utf8') as wf:
            wf.write(str(i) + "||" + keywords_texts[i] + "||" + sku_names_texts[idx] + "\n")

def get_results(results):
    filename = "result/bm25_v1_results.txt"
    with open(filename, 'w') as wf:
        try:
            for line in results:
                wf.write(line + "\n")
        except:
            print("Write files error!!!")

if __name__ == '__main__':
    train()