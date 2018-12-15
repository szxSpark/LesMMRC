import json
max_article_len = 1200
max_question_len = 142
def get_plain_text(path, article_path, question_path):
    print("load ", path)
    data = json.load(open(path, "r", encoding="utf-8"))
    article_list = []
    question_list = []
    for article_id, d in enumerate(data):
        temp_article = list(d['article_content'])
        temp_article = temp_article[:max_article_len]
        for qa in d['questions']:
            if len(qa['question']) != 0 and len(qa['answer']) != 0 \
                and qa['answer_span'][0] != -1 and qa['answer_span'][1] != -1\
                and qa['answer_span'][1] < max_article_len:    # TODO
                article_list.append(temp_article)
                temp_question = list(qa['question'])
                temp_question = temp_question[:max_question_len]
                question_list.append(temp_question)
    print(len(article_list), len(question_list))
    wf1 = open(article_path, "w", encoding="utf-8")
    for line in article_list:
        wf1.write("\t".join(line) + "\n")
    wf2 = open(question_path, "w", encoding="utf-8")
    for line in question_list:
        wf2.write("\t".join(line) + "\n")


def get_test_plain_text(path, article_path, question_path):
    print("load ", path)
    data = json.load(open(path, "r", encoding="utf-8"))
    article_list = []
    question_list = []
    for _, d in enumerate(data):
        temp_article = list(d['article_content'])
        temp_article = temp_article[:max_article_len]
        for qa in d['questions']:
            # 问题答案对
            article_list.append(temp_article)
            temp_question = list(qa['question'])
            temp_question = temp_question[:max_question_len]
            question_list.append(temp_question)

    print(len(article_list), len(question_list))
    wf1 = open(article_path, "w", encoding="utf-8")
    for line in article_list:
        wf1.write("\t".join(line) + "\n")
    wf2 = open(question_path, "w", encoding="utf-8")
    for line in question_list:
        wf2.write("\t".join(line) + "\n")

# data_path = "/disk/lhliu/szx_project/junshizhineng/train_data/question_with_span_seg.json"
# article_path = "/disk/lhliu/szx_project/junshizhineng/train_data/article_plain.txt"
# question_path = "/disk/lhliu/szx_project/junshizhineng/train_data/question_plain.txt"
# get_plain_text(data_path, article_path, question_path)

data_path = "/disk/lhliu/szx_project/junshizhineng/train_data/chusai_preprocessed_seg.json"
article_path = "/disk/lhliu/szx_project/junshizhineng/test_data/article_plain.txt"
question_path = "/disk/lhliu/szx_project/junshizhineng/test_data/question_plain.txt"
get_test_plain_text(data_path, article_path, question_path)