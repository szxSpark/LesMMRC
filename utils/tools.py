from collections import Counter

num = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
kin = ['十', '百', '千', '万', '零']

def sadd(x):
    x.reverse()
    if len(x) >= 2:
        x.insert(1, kin[0])
        if len(x) >= 4:
            x.insert(3, kin[1])
            if len(x) >= 6:
                x.insert(5, kin[2])
                if len(x) >= 8:
                    x.insert(7, kin[3])
                    if len(x) >= 10:
                        x.insert(9, kin[0])
                        if len(x) >= 12:
                            x.insert(11, kin[1])

    x = fw(x)
    x = d1(x)
    x = d2(x)
    x = dl(x)
    return x

def d1(x):
    if '零' in x:
        a = x.index('零')
        if a == 0:
            del x[0]
            d1(x)
        else:
            if x[a + 2] in ['十', '百', '千', '万', '零']:
                if x[a + 1] != '万':
                    del x[a + 1]
                    d1(x)
    return x

def d2(x):
    try:
        a = x.index('零')
        if x[a - 1] in ['十', '百', '千', '零']:
            del x[a - 1]
            d2(x[a + 1])
    except:
        pass
    return x

def fw(x):
    if len(x) >= 9:
        if x[8] == '零':
            del x[8]
    return x

def dl(x):
    try:
        if x[0] == '零':
            del x[0]
    except:
        pass
    x.reverse()
    x = ''.join(x)
    return x

def convert(number):
    if number == 0:
        return "零"

    number = list(str(number))
    for j in number:
        number[(number.index(j))] = num[int(j)]
    number = sadd(number)
    if len(number) >= 2 and number[:2] == "一十":
        number = "十"+number[2:]
    return number

def precision_recall_f1(prediction, ground_truth):
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def recall(prediction, ground_truth):
    """
    This function calculates and returns the recall
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of recall
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[1]

def f1_score(prediction, ground_truth):
    """
    This function calculates and returns the f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of f1
    Raises:
        None
    """
    return precision_recall_f1(prediction, ground_truth)[2]

def metric_max_over_ground_truths(metric_fn, prediction, ground_truth):
    score = metric_fn(list(prediction), list(ground_truth))
    return score

def sentences_splitted(text, split_chars=["。"]):
    # text : list, 1-dim
    # 按照分隔符进行分句

    splitted = []
    idxs = [i for i, a in enumerate(text) if a in split_chars]
    if len(idxs) == 0:
        return [text]
    for i, _ in enumerate(idxs):
        if i == 0:
            splitted.append(text[:idxs[i] + 1])
        else:
            splitted.append(text[idxs[i - 1] + 1: idxs[i] + 1])

    if idxs[-1] != len(text) - 1:
        splitted.append(text[idxs[-1] + 1:])

    return splitted

def find_answer_span(article, question, answer):
    '''

    :param article: str
    :param question: str
    :param answer: str
    :return:
    '''
    best_match_score = 0
    best_match_span = [-1, -1]
    best_fake_answer = None
    best_sent_idx = 0
    answer_tokens = set(answer)

    # article -> sent_list
    answer_sent_num = len([c for c in list(answer) if c == "。"])
    if answer[-1] != '。':
        answer_sent_num += 1
    sent_list = sentences_splitted(article)
    sent_len_dict = {}
    for i, sent in enumerate(sent_list):
        sent_len_dict[i] = len(sent)

    if answer_sent_num > 1:
        new_sent_list = []
        for i, _ in enumerate(sent_list):
            if i > len(sent_list) - answer_sent_num:
                break
            else:
                new_sent_list.append("".join(sent_list[i:i + answer_sent_num]))
        assert len(new_sent_list) != 0
        sent_list = new_sent_list[:]
    for i, sent in enumerate(sent_list):
        # sent: str
        question_match_score = metric_max_over_ground_truths(recall, sent, question)
        for start_tidx in range(len(sent)):
            if sent[start_tidx] not in answer_tokens:
                continue
            for end_tidx in range(len(sent) - 1, start_tidx - 1, -1):
                span_tokens = sent[start_tidx: end_tidx + 1]
                match_score = metric_max_over_ground_truths(f1_score, span_tokens, answer)
                match_score = 0.5 * (match_score + question_match_score)
                if match_score >= best_match_score:
                    best_match_span = [start_tidx, end_tidx]
                    best_match_score = match_score
                    best_sent_idx = i
                    best_fake_answer = ''.join(span_tokens)
    pre = sum([sent_len_dict[i] for i in range(best_sent_idx)])
    best_match_span = [pre+s for s in best_match_span]
    # print(article[best_match_span[0]:best_match_span[1]+1])
    return best_match_score, best_match_span

def DBC2SBC(ustring):
    # ' 全角转半角 ”'
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
        rstring += chr(inside_code)
    return rstring

def SBC2DBC(ustring):
    #半角转全角
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x0020:
            inside_code = 0x3000
        else:
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
            inside_code += 0xfee0
        rstring += chr(inside_code)
    return rstring