import os
import json
LTP_DATA_DIR = '../../train_data/ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
from pyltp import Segmentor

def seg(input_file, output_file):
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型

    data = json.load(open(input_file, "r", encoding="utf-8"))
    count = 0
    for d in data:
        count += 1
        print(count)
        d['article_content'] = list(segmentor.segment(d['article_content']))
        d['article_title'] = list(segmentor.segment(d['article_title']))

        questions = d["questions"]
        for q in questions:
            q['answer'] = list(segmentor.segment(q['answer']))  # 其实没什么用，现在是用answer_span进行计算
            q['question'] = list(segmentor.segment(q['question']))
            # 根据字级别的answer_span 生成词语级别的
            if len(q['answer']) == 0 or len(q['question']) == 0:
                continue
            answer_span_char_level = q['answer_span']
            answer_span_word_level = answer_span_char2word(d['article_content'], answer_span_char_level)
            q['answer_span'] = answer_span_word_level

    segmentor.release()  # 释放模型
    json.dump(data, open(output_file, "w", encoding="utf-8"), ensure_ascii=False)

def answer_span_char2word(article, answer_span_char_level):
    # article: 分词后的list
    if answer_span_char_level[0] == -1 and answer_span_char_level[1] == -1:
        return [-1, -1]
    count = 0
    char_start = answer_span_char_level[0]+1
    char_end = answer_span_char_level[1]+1
    word_start, word_end = -1, -1
    for idx, word in enumerate(article):
        count += len(word)
        # print(idx, word, count, char_start, char_end)
        if count >= char_start and word_start == -1:
            word_start = idx
        if count >= char_end and word_end == -1:
            word_end = idx
    return [word_start, word_end]
