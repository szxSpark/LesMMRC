#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
from tools import convert, find_answer_span, DBC2SBC
from seg import seg

def gen_answer_span(article, question, answer):
    # article : str
    # question: str
    # answer  : str

    # 全角替换半角，但是不写入json数据
    article = DBC2SBC(article)
    question = DBC2SBC(question)
    answer = DBC2SBC(answer)

    answer_list = [answer]
    if str(answer).isdigit():
        if answer == "2":
            answer_list.append("两")
        answer_list.append(convert(int(answer)))

    answer_span = None
    tmp_score = -1
    for ans in answer_list:
        # 根据 article question ans 匹配最高相似度的sent
        match_score, match_span = find_answer_span(article=article, question=question, answer=ans)
        if match_score >= tmp_score:
            answer_span = match_span
            tmp_score = match_score
        # 根据答案得到并列第一的sent_list，用question重新计算F值
    return answer_span

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
            ans = "".join(q['answer'].split())
            que = "".join(q['question'].split())
            q['answer'] = ans
            q['question'] = que
            if len(ans) == 0 or len(que) == 0:
                continue
            # ---
            # if q['answer_span'][0] < 0 or q['answer_span'][1] < 0:
            #     # print(d)
            #     # print(article)
            #     # print(ans)
            #     article = d['article_title']+"。"+article
            #     answer_span = gen_answer_span(article, que, ans)
            #     print(answer_span, q['answer_span'])
            #     if answer_span[0]==-1:
            #         # 这里是抽象型回答，或标注错误的回答。没有办法生成answer_span，一共48个
            #         pass
            #     q['answer_span'] = answer_span
            # ---
            answer_span = gen_answer_span(article, que, ans)
            q['answer_span'] = answer_span
    json.dump(data, open(output_file, "w", encoding='utf-8'), ensure_ascii=False)

if __name__ == "__main__":
    # 1. 预处理，修整原文不正常标点符号，保留原文的字符，将标题与原文拼接到一起
    # 2. 生成字级别的answer_span，这样可以避免分词错误
    # main1(input_file="../../train_data/question.json", output_file="../../train_data/question_with_span.json")

    # 3. 对question_with_span 的 article question answer进行分词处理。
    # 4. 针对字级别的answer_span 设计算法，生成词级别的answer_span
    seg(input_file="../../train_data/question_with_span.json", output_file="../../train_data/question_with_span_seg.json")