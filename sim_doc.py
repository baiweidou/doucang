#-*-codeing = utf-8 -*-
#@Time :2020/9/15 22:25
#@Author : baiweidou
#@File : Edit distance calculation.py
#@Software :PyCharm
#第一份编程作业
import jieba
import sys
import time
from gensim import corpora,models,similarities

#文档分句，删除符号，保留中文
def creat_sentence(file_data):
    file_sentence = []
    s = ''
    for word in file_data:
        #中文编码范围
        if '\u4e00'<=word<='\u9fff':
            s += word
        #逗号分句
        elif word == '，':
            file_sentence.append(s)
            s = ''
    return file_sentence
#
def tfidf_model(orig_items,orig_sim_items):
    #生成词典
    dictionary = corpora.Dictionary(orig_items)
    #生成稀疏向量库
    corpus = [dictionary.doc2bow(text) for text in orig_items]
    #利用TFidf模型建模
    tf = models.TfidfModel(corpus)
    #通过token2id得到特征数（字典里面的键的个数）
    num_features = len(dictionary.token2id.keys())
    #建立索引
    index = similarities.MatrixSimilarity(tf[corpus],num_features=num_features)
    #索引持久性
    index.save('index.txt')

    #求相似度列表
    sim_value = []
    for i in range(0,len(orig_sim_items)):
        orig_sim_vec = dictionary.doc2bow(orig_sim_items[i])
        sim = index[tf[orig_sim_vec]]
    #取相似度最大的值
        sim_max = max(sim)
        sim_value.append(sim_max)
    return sim_value

#求每段话在文章中的权重
def get_weight(file_data,file_sentence):
    #权重列表
    weight = []
    #计算文章总字词长度
    file_len = 0
    #w用来留存每句话的权重
    w = 0
    for word in file_data:
        if '\u4e00'<=word<='\u9fff':
            file_len +=1
    for sentence in file_sentence:
        for word in sentence:
            if '\u4e00' <= word <= '\u9fff':
                w += 1
        weight.append(w/file_len)
        w = 0
    return weight
if __name__ == '__main__':
    time_start = time.time()
    #原始文档打开
    orig_file = open(sys.argv[1],'r', encoding='UTF-8')
    #相似文档打开
    orig_sim_file = open(sys.argv[2],'r',encoding='utf-8')
    #原始文档写入
    orig_data = orig_file.read()
    #相似文档写入
    orig_sim_data = orig_sim_file.read()
    #原始文档相似文档关闭
    orig_file.close()
    orig_sim_file.close()
    #原始文档分句分词
    orig_sentence =creat_sentence(orig_data)
    orig_word = [[word for word in jieba.lcut(sentence)] for sentence in orig_sentence]
    #相似文档分句分词
    orig_sim_sentence = creat_sentence(orig_sim_data)
    orig_sim_word = [[word for word in jieba.lcut(sentence)] for sentence in orig_sim_sentence]
    #获取相似文档权重列表
    weight = get_weight(orig_sim_data,orig_sim_sentence)
    #获取相似文档相似度列表
    sim_value = tfidf_model(orig_word,orig_sim_word)
    #ans用于存放最后总相似度，为每句权重和相似度的积
    ans = 0
    for i in range(len(weight)):
        ans += weight[i]*sim_value[i]
    #保留两位数
    ans = (str("%.2f") % ans)
    print(ans)
    #写入文件
    file = open('test_ans.txt','w', encoding='UTF-8')
    file.write(ans)
    file.close()
    #计时
    time_end = time.time()
    time = time_end-time_start
    print(time)