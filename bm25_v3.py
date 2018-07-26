# -*- coding: utf-8 -*-
"""
    BM25_v3 改进版
"""
import jieba
import pandas as pd
import bm25
import string
import re
from zhon.hanzi import punctuation
from gensim import corpora, models, similarities

# jieba添加用户词典(一些型号词)
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

def get_stop_words():
    """
        得到停用词表
    :return:
    """
    stop_words_file = 'stop_words.txt'
    stop_words = open(stop_words_file, 'r', encoding='utf8').readlines()
    stop_words = [sw.strip() for sw in stop_words]
    stop_words_dict = {}
    for sw in stop_words:
        if sw not in stop_words_dict:
            stop_words_dict[sw] = len(stop_words_dict)
    return stop_words_dict

def get_text_jieba(texts):
    '''
        得到jieba分词结果
    :param train_texts:
    :return:
    '''
    stop_words_dict = get_stop_words()
    print("Stop words number:", len(stop_words_dict))

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
            if len(word) > 0 and word not in stop_words_dict.keys():
                doc_list.append(word)
        all_doc_list.append(doc_list)

    return all_doc_list

def train():
    '''
        匹配过程
    :return:
    '''
    # 构建匹配语料库 398872 samples
    sku_names_texts = get_train_datas()
    sku_names_jieba = get_text_jieba(sku_names_texts)
    print("匹配语料库数据规模：")
    print(len(sku_names_texts), len(sku_names_jieba))

    # 测试数据 1000 samples
    keywords_texts = get_test_datas()
    keywords_jieba = get_text_jieba(keywords_texts)
    print("测试数据集规模：")
    print(len(keywords_texts))

    # 统计词表
    dictionary = corpora.Dictionary(sku_names_jieba)
    print("词典长度：")
    print(len(dictionary))

    # 用gensim建立BM25模型
    bm25Model = bm25.BM25(sku_names_jieba)
    # 根据gensim源码，计算平均逆文档频率
    average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())

    for i, item in enumerate(keywords_jieba):
        scores = bm25Model.get_scores(item, average_idf)
        # sorted_scores = sorted(scores, reverse=True)[:10]
        idx = scores.index(max(scores))
        print(i, "||", keywords_texts[i], "||", sku_names_texts[idx])

        with open("result/bm25_v3_results.txt", 'a', encoding='utf8') as wf:
            wf.write(str(i) + "||" + keywords_texts[i] + "||" + sku_names_texts[idx] + "\n")

def get_results(results):
    filename = "result/bm25_v3_results.txt"
    with open(filename, 'w') as wf:
        try:
            for line in results:
                wf.write(line + "\n")
        except:
            print("Write files error!!!")

if __name__ == "__main__":
    train()