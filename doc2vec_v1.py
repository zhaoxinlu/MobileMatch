# -*- coding: utf-8 -*-
import jieba
import pandas as pd
import string
import re
import multiprocessing
from zhon.hanzi import punctuation
from gensim import corpora, models, similarities
from gensim.models import doc2vec

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

        doc_list = []
        for word in jieba.cut(doc.strip()):
            if len(word) > 0 and word not in stop_words_dict.keys() and word != ' ':
                doc_list.append(word)
        all_doc_list.append(doc_list)

    return all_doc_list

def train():
    stop_words_dict = get_stop_words()
    sku_names_texts = get_train_datas()
    sku_names_jieba = []
    for i, doc in enumerate(sku_names_texts):
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

        document = doc2vec.TaggedDocument(words=doc_list, tags=[i])
        sku_names_jieba.append(document)

    keywords_texts = get_test_datas()
    keywords_jieba = get_text_jieba(keywords_texts)

    model = doc2vec.Doc2Vec(vector_size=300, min_count=1, epochs=20)
    model.build_vocab(sku_names_jieba)
    model.train(sku_names_jieba, total_examples=model.corpus_count, epochs=model.epochs)

    inferred_vector = model.infer_vector(keywords_jieba[35])
    print(keywords_jieba[35])
    sims = model.docvecs.most_similar([inferred_vector], topn=1)
    idx = sims[0][0]

    print(sku_names_texts[idx])

if __name__ == '__main__':
    train()