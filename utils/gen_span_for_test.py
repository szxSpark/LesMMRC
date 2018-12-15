#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import os
LTP_DATA_DIR = '../../train_data/ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
from pyltp import Segmentor

def preprocess_article(article):
    # 分割段落、标题标识符\u3000
    article = "。".join(article.split())
    # 修补分割错误
    idxs = [i for i, a in enumerate(article) if a == "。"]
    chars = list("：；？！，。（）【】")
    for idx in idxs[::-1]:
        if idx != 0:
            if article[idx - 1] in chars:
                article = article[:idx] + article[idx + 1:]
    # 此处的article是一个可以直接使用的字符串
    return article

def main1(input_file, output_file):
    '''
    对最粗糙的数据进行字级别的预处理,不对原文中的字词产生任何更改，
    然后生成字级别的answer_span，以半角为原则
    '''
    data = json.load(open(input_file, "r", encoding="utf-8"))
    # list: [dict],
    # questions, list -> [dict] -> question_type, question, answer, questions_id
    # article_title, str
    # article_id, str
    # article_type, str
    # article_content, str
    count = 0
    for d in data:
        count += 1
        print(count)
        article = d['article_content']
        article = "".join(d['article_title'].split()) + "。" + article  # 拼接标题
        article = preprocess_article(article)
        d['article_content'] = article
        questions = d["questions"]
        for q in questions:
            que = "".join(q['question'].split())
            q['question'] = que

    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    count = 0
    for d in data:
        count += 1
        print(count)
        d['article_content'] = list(segmentor.segment(d['article_content']))
        d['article_title'] = list(segmentor.segment(d['article_title']))

        questions = d["questions"]
        for q in questions:
            q['question'] = list(segmentor.segment(q['question']))
    segmentor.release()  # 释放模型
    json.dump(data, open(output_file, "w", encoding="utf-8"), ensure_ascii=False)

if __name__ == "__main__":
    # 1. 预处理，修整原文不正常标点符号，保留原文的字符，将标题与原文拼接到一起
    # 2. 生成字级别的answer_span，这样可以避免分词错误
    # 3. 对question_with_span 的 article question answer进行分词处理。
    # 4. 针对字级别的answer_span 设计算法，生成词级别的answer_span
    main1(input_file="../../train_data/chusai.json", output_file="../../train_data/chusai_preprocessed_seg.json")