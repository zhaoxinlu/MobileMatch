# -*- coding: utf-8 -*-
"""
    gensim word2vec
"""
import re
import jieba
import string
import pandas as pd
import multiprocessing
from zhon.hanzi import punctuation
from gensim.models import Word2Vec

# jieba添加用户词典
user_words_file = 'user_words.txt'
user_words = open(user_words_file, 'r', encoding='utf8').readlines()
user_words = [uw.strip() for uw in user_words]
for uw in user_words:
    jieba.add_word(uw)

def get_sku_names():
    datas = pd.read_csv("data/mobile.csv")
    sku_names = datas["sku_name"]
    return sku_names

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
            if len(word) > 0 and word != ' ' and word not in stop_words_dict.keys():
                doc_list.append(word)
        all_doc_list.append(doc_list)

    return all_doc_list

def train_w2v():
    """
        训练skuname词向量
    :return:
    """
    sku_names_texts = get_sku_names()
    sku_names_jieba = get_text_jieba(sku_names_texts)

    print("Training w2v...")
    sku_names_model = Word2Vec(sku_names_jieba, min_count=5, workers=multiprocessing.cpu_count(), size=100, sg=1)
    sku_names_model.save('models/skuname100.model')
    sku_names_model.wv.save_word2vec_format('models/skuname100.vector', binary=False)
    print("Train and save done!")

if __name__ == '__main__':
    train_w2v()