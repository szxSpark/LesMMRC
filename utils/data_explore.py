#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
def length_no_seg_no_preprocess():
    # path = "/Users/zxsong/Downloads/nlp-data/ReadingComprehension-JunShiZhiNeng/question.json"
    path = "../../train_data/question.json"
    data = json.load(open(path, "r", encoding="utf-8"))
    # list: [dict],

    # questions, list -> [dict] -> question_type, question, answer, questions_id
    # article_title, str
    # article_id, str
    # article_type, str
    # article_content, str
    count = 0
    article_len = []
    question_len = []
    answer_len = []
    for d in data:
        count += 1
        # print(count)
        article = d['article_content']
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
        article_len.append(len(article))
        questions = d["questions"]
        for q in questions:
            ans = "".join(q['answer'].split())
            que = "".join(q['question'].split())
            answer_len.append(len(ans))
            question_len.append(len(que))
    analysis_len(article_len)
    analysis_len(question_len, split=20)
    print(max(question_len))  # 242
    analysis_len(answer_len, split=20)
    print(max(answer_len))  # 255

def length_seg_preprocess():
    path = "../../train_data/chusai_preprocessed_seg.json"
    # path = "../question.json"
    data = json.load(open(path, "r", encoding="utf-8"))
    # list: [dict],

    # questions, list -> [dict] -> question_type, question, answer, questions_id
    # article_title, str
    # article_id, str
    # article_type, str
    # article_content, str
    count = 0
    article_len = []
    question_len = []
    for d in data:
        count += 1
        # print(count)
        article = d['article_content']
        article_len.append(len(article))
        questions = d["questions"]
        for q in questions:
            question_len.append(len(q['question']))
    analysis_len(article_len)
    analysis_len(question_len, split=20)
    print(max(question_len))  # 242

def length_seg_answer_span():
    path = "../../train_data/question_with_span_seg.json"
    # path = "../question.json"
    data = json.load(open(path, "r", encoding="utf-8"))
    # list: [dict],

    # questions, list -> [dict] -> question_type, question, answer, questions_id
    # article_title, str
    # article_id, str
    # article_type, str
    # article_content, str
    count = 0
    max_answer_span_len = []
    for d in data:
        # print(count)
        questions = d["questions"]
        for q in questions:
            if len(q['answer']) != 0:
                max_answer_span_len.append(q['answer_span'][1])
                if q["answer_span"][0] == -1 and q["answer_span"][1] == -1:
                    count += 1
    print(count)
    # analysis_len(max_answer_span_len, split=500)
    # print(max(max_answer_span_len))
    # print(min(max_answer_span_len))


def analysis_len(data_list, split=500):
    tmp_dict = {}
    for d in data_list:
        length = d
        k = str(length // split * split) + "~" + str((length // split + 1) * split)
        if k not in tmp_dict:
            tmp_dict[k] = 1
        else:
            tmp_dict[k] += 1
    print(sorted(tmp_dict.items(), key=lambda x: int(x[0].split("~")[0]), reverse=False))


if __name__ == "__main__":
    # 输入的文档可以是answer_span附近固定长度的文档
    # answer_span要重新计算

    # length_no_seg_no_preprocess()
    length_seg_preprocess()

    # length_seg_answer_span()
    # 抽象型回答，或标注错误的回答。没有办法生成answer_span，一共48个
'''
length_no_seg_no_preprocess
[('0~500', 6465), ('500~1000', 6704), ('1000~1500', 2998), ('1500~2000', 1457), ('2000~2500', 697), ('2500~3000', 444), ('3000~3500', 319), ('3500~4000', 233), ('4000~4500', 175), ('4500~5000', 109), ('5000~5500', 81), ('5500~6000', 77), ('6000~6500', 50), ('6500~7000', 29), ('7000~7500', 28), ('7500~8000', 24), ('8000~8500', 18), ('8500~9000', 7), ('9000~9500', 10), ('9500~10000', 17), ('10000~10500', 4), ('10500~11000', 3), ('11000~11500', 4), ('11500~12000', 3), ('12500~13000', 5), ('14000~14500', 1), ('14500~15000', 1), ('15000~15500', 2), ('16000~16500', 1), ('16500~17000', 2), ('19500~20000', 1), ('20000~20500', 1), ('21000~21500', 1), ('21500~22000', 1), ('22000~22500', 1), ('24000~24500', 1), ('26000~26500', 1), ('32000~32500', 1), ('36000~36500', 1), ('36500~37000', 1), ('37000~37500', 1), ('37500~38000', 1), ('41000~41500', 1), ('41500~42000', 1), ('43000~43500', 1), ('43500~44000', 1), ('44500~45000', 1), ('49500~50000', 2), ('51500~52000', 1), ('54000~54500', 1), ('54500~55000', 1), ('59500~60000', 1), ('60000~60500', 1), ('61500~62000', 3), ('63500~64000', 1), ('64000~64500', 1), ('66500~67000', 1), ('76500~77000', 1), ('81500~82000', 1)]
[('0~20', 34906), ('20~40', 32290), ('40~60', 16668), ('60~80', 8258), ('80~100', 3471), ('100~120', 1434), ('120~140', 584), ('140~160', 216), ('160~180', 112), ('180~200', 32), ('200~220', 19), ('220~240', 5), ('240~260', 2)]
242
[('0~20', 79805), ('20~40', 11221), ('40~60', 3504), ('60~80', 1582), ('80~100', 742), ('100~120', 428), ('120~140', 230), ('140~160', 185), ('160~180', 115), ('180~200', 68), ('200~220', 62), ('220~240', 39), ('240~260', 16)]
255

length_seg_preprocess()
训练集
[('0~500', 11918), ('500~1000', 4965), ('1000~1500', 1472), ('1500~2000', 673), ('2000~2500', 389), ('2500~3000', 209), ('3000~3500', 134), ('3500~4000', 74), ('4000~4500', 37), ('4500~5000', 37), ('5000~5500', 17), ('5500~6000', 20), ('6000~6500', 7), ('6500~7000', 6), ('7000~7500', 4), ('8000~8500', 1), ('8500~9000', 3), ('9000~9500', 1), ('9500~10000', 1), ('11000~11500', 2), ('12000~12500', 1), ('12500~13000', 2), ('13500~14000', 1), ('14000~14500', 1), ('18500~19000', 1), ('20500~21000', 1), ('21000~21500', 1), ('22000~22500', 1), ('24000~24500', 2), ('24500~25000', 1), ('25500~26000', 1), ('27500~28000', 1), ('28500~29000', 3), ('29000~29500', 1), ('31500~32000', 1), ('32000~32500', 1), ('33500~34000', 1), ('34000~34500', 2), ('34500~35000', 1), ('35500~36000', 1), ('36000~36500', 1), ('36500~37000', 1), ('42000~42500', 1), ('45000~45500', 1), ('45500~46000', 1)]
[('0~20', 61798), ('20~40', 27066), ('40~60', 7109), ('60~80', 1604), ('80~100', 331), ('100~120', 72), ('120~140', 15), ('140~160', 2)]
141
[('0~20', 89591), ('20~40', 5784), ('40~60', 1555), ('60~80', 556), ('80~100', 273), ('100~120', 130), ('120~140', 80), ('140~160', 24), ('160~180', 4)]
168

初赛测试集
[('0~500', 10510), ('500~1000', 5570), ('1000~1500', 1850), ('1500~2000', 874), ('2000~2500', 490), ('2500~3000', 270), ('3000~3500', 164), ('3500~4000', 79), ('4000~4500', 59), ('4500~5000', 31), ('5000~5500', 26), ('5500~6000', 13), ('6000~6500', 10), ('6500~7000', 9), ('7000~7500', 2), ('7500~8000', 3), ('8000~8500', 3), ('8500~9000', 2), ('9000~9500', 4), ('9500~10000', 1), ('10500~11000', 1), ('11500~12000', 2), ('12500~13000', 3), ('13500~14000', 1), ('16000~16500', 1), ('24000~24500', 1), ('25000~25500', 2), ('25500~26000', 1), ('27000~27500', 1), ('29500~30000', 1), ('30000~30500', 1), ('31500~32000', 1), ('32000~32500', 1), ('34000~34500', 2), ('35000~35500', 2), ('36500~37000', 2), ('38000~38500', 2), ('38500~39000', 1), ('41000~41500', 1), ('43000~43500', 2), ('44000~44500', 1)]
[('0~20', 68761), ('20~40', 27231), ('40~60', 4046), ('60~80', 490), ('80~100', 71), ('100~120', 14), ('120~140', 8)]
'''