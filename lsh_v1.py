# -*- coding: utf-8 -*-
"""
    LSH Forest--LSH+随机投影树
    思想：利用随机投影树，对所有的数据进行划分，将每次搜索与计算的点的数目减小到一个可接受的范围，
    然后建立多个随机投影树构成随机投影森林，将森林的综合结果作为最终的结果。
"""
import jieba
import pandas as pd
import string
from zhon.hanzi import punctuation
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LSHForest

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

        if doc == '':
            doc = "手机错误"

        doc_list = []
        for word in jieba.cut(doc.strip()):
            if len(word) > 0 and word not in stop_words_dict.keys() and word != ' ':
                doc_list.append(word)
        all_doc_list.append(doc_list)

    return all_doc_list

def train():
    # 构建匹配语料库 398872 samples
    sku_names_texts = get_train_datas()
    sku_names_jieba = get_text_jieba(sku_names_texts)
    sku_names_with_spaces = []
    for sku_names in sku_names_jieba:
        sku_names_with_spaces.append(' '.join(sku_names))

    # 测试数据 1000 samples
    keywords_texts = get_test_datas()
    keywords_jieba = get_text_jieba(keywords_texts)
    keywords_with_spaces = []
    for keywords in keywords_jieba:
        keywords_with_spaces.append(' '.join(keywords))

    tfidf_vec = TfidfVectorizer(min_df=3, max_features=None, ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1)
    x_train = tfidf_vec.fit_transform(sku_names_with_spaces)

    lshf = LSHForest(random_state=42)
    #lshf.fit(np.array(x_train))
    lshf.fit(x_train)

    for i, kw in enumerate(keywords_with_spaces):
        x_test = tfidf_vec.transform([kw])
        distances, indices = lshf.kneighbors(x_test.toarray(), n_neighbors=1)
        idx = indices[0][0]
        print(i, "||", keywords_texts[i], "||", sku_names_texts[idx])

        with open("result/lsh_v1_results.txt", 'a', encoding='utf8') as wf:
            wf.write(str(i) + "||" + keywords_texts[i] + "||" + sku_names_texts[idx] + "\n")

if __name__ == "__main__":
    train()