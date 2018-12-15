#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
from collections import OrderedDict
from model.bidaf import BiDAF
# from model.llh_slqa import SLQA  # singmodel elmo
from model.llh_slqa2 import SLQA
import torch
import argparse
from data.buil_dataset import Data
import torch.nn.functional as F
import numpy as np
import h5py
from tqdm import tqdm
import pickle

def identiy_result():
    path = "/Users/zxsong/Downloads/nlp-data/ReadingComprehension-JunShiZhiNeng/question.json"
    data = json.load(open(path, "r", encoding="utf-8"))
    result = []
    for d in data:
        tmp = OrderedDict()
        tmp["article_id"] = d["article_id"]
        questions = d["questions"]
        tmp["questions"] = []
        for q in questions:
            tmp["questions"].append({
                "questions_id": q["questions_id"],
                "answer": q["answer"]
            })
        result.append(tmp)
    print(len(result))
    json.dump(result, open("indentity_result.json", "w", encoding="utf-8"), ensure_ascii=False)

def fake_answer_span_result():
    path = "./question_with_span_seg.json"
    data = json.load(open(path, "r", encoding="utf-8"))
    result = []
    for d in data:
        tmp = OrderedDict()
        tmp["article_id"] = d["article_id"]
        questions = d["questions"]
        tmp["questions"] = []
        article = d['article_content']
        for q in questions:
            if "answer_span" in q:
                answer = article[q["answer_span"][0]:q["answer_span"][1] + 1]
            else:
                print("no answer")
                answer = ""
            tmp["questions"].append({
                "questions_id": q["questions_id"],
                "answer": "".join(list(answer))
            })
        result.append(tmp)
    print(len(result))
    json.dump(result, open("fake_answer_span_result.json", "w", encoding="utf-8"), ensure_ascii=False)

def predict_one_batch_size(args):
    print("load ", args.train_file)
    data = json.load(open(args.train_file, "r", encoding="utf-8"))
    result = []
    word2idx = json.load(open(args.word2idx_path, "r", encoding="utf-8"))
    args.word_vocab_size = len(word2idx)
    print("word_vocab_size:", args.word_vocab_size)

    def word2idx_func(temp_list):
        return [word2idx[w] if w in word2idx else 0 for w in temp_list]

    # model = BiDAF(args)
    model = SLQA(args)

    print("load model data:", args.model_path)
    model.load_state_dict(torch.load(args.model_path))
    if args.cuda:
        model = model.cuda()
    count = 0
    for d in data:
        count +=1
        print(count)
        tmp = OrderedDict()
        tmp["article_id"] = d["article_id"]
        tmp["questions"] = []

        raw_article = d["article_content"]
        questions = d["questions"]
        for q in questions:
            raw_question = q['question']
            question = word2idx_func(raw_question)
            article = word2idx_func(raw_article)
            if len(question) == 0:
                tmp["questions"].append({
                    "questions_id": q["questions_id"],
                    "answer": ""  # TODO
                })
                continue
            if len(article) > 1300000 //len(question):
                # OOM
                article = article[:1300000//len(question)]
            print("len(article):", len(article), "len(question)", len(question), " len(article)*len(question)=", len(article)*len(question))
            article = torch.LongTensor(article)
            article = torch.unsqueeze(article, 0)
            question = torch.LongTensor(question)
            question = torch.unsqueeze(question, 0)

            if args.cuda:
                article, question = article.cuda(), question.cuda()
            print(article.size(), question.size())
            p1, p2 = model(article, question)
            _, p1_predicted = torch.max(p1, 1)
            _, p2_predicted = torch.max(p2, 1)
            p1_predicted = p1_predicted.item()
            p2_predicted = p2_predicted.item()

            tmp["questions"].append({
                "questions_id": q["questions_id"],
                "answer": "".join(raw_article[p1_predicted:p2_predicted+1])
            })
        result.append(tmp)
    print(len(result))
    json.dump(result, open(args.result_file, "w", encoding="utf-8"), ensure_ascii=False)

def predict(args):
    db = Data(args)
    db.load_vocab()
    db.build_test_dataset()
    # model = BiDAF(args)
    model = SLQA(args)

    print("load model data:", args.model_path)
    model.load_state_dict(torch.load(args.model_path))
    if args.cuda:
        model = model.cuda()
    q2a = {}
    q2idx = {}
    result = []
    for i, d in enumerate(db.data):
        tmp = OrderedDict()
        tmp["article_id"] = d["article_id"]
        tmp["questions"] = []
        for q in d["questions"]:
            q2a[q["questions_id"]] = d["article_id"]
            q2idx[q["questions_id"]] = i
        result.append(tmp)
    raw_article_list = db.test_raw_article_list
    article_id_list = db.test_article_id_list
    question_id_list = db.test_question_id_list
    idx = 0
    for article, question,  _, in tqdm(db.test_loader):
        if args.cuda:
            article, question = article.cuda(), question.cuda()
        p1, p2 = model(article, question, is_training=False)
        _, p1_predicted = torch.max(p1, 1)
        _, p2_predicted = torch.max(p2, 1)
        p1_predicted = p1_predicted.cpu().numpy().tolist()
        p2_predicted = p2_predicted.cpu().numpy().tolist()
        for _p1, _p2, _raw_article, _article_id, _question_id in zip(p1_predicted, p2_predicted,
                                                                     raw_article_list[idx:idx+len(p1_predicted)],
                                                                     article_id_list[idx:idx+len(p1_predicted)],
                                                                     question_id_list[idx:idx + len(p1_predicted)]):
            assert q2a[_question_id] == _article_id
            result[q2idx[_question_id]]["questions"].append({
                "questions_id": _question_id,
                "answer": "".join(_raw_article[_p1:_p2 + 1])
            })
        idx = idx + len(p1_predicted)

    print(len(result))
    json.dump(result, open(args.result_file, "w", encoding="utf-8"), ensure_ascii=False)

def predict_multi_model(args):
    db = Data(args)
    db.load_vocab()
    db.build_test_dataset()
    # model = BiDAF(args)
    model_path_list = args.model_path_list

    q2a = {}
    q2idx = {}
    result = []
    for i, d in enumerate(db.data):
        tmp = OrderedDict()
        tmp["article_id"] = d["article_id"]
        tmp["questions"] = []
        for q in d["questions"]:
            q2a[q["questions_id"]] = d["article_id"]
            q2idx[q["questions_id"]] = i
        result.append(tmp)
    raw_article_list = db.test_raw_article_list
    article_id_list = db.test_article_id_list
    question_id_list = db.test_question_id_list

    p1, p2 = [], []
    model = SLQA(args)
    if args.cuda:
        model = model.cuda()
    print(args.max_article_len)
    for model_id, model_path in enumerate(model_path_list):
        print(args.batch_size)
        print(model_id, "load model data:", model_path)
        model.load_state_dict(torch.load(model_path))
        saved_p1 = []
        saved_p2 = []
        batch_idx = 0
        for article, question,  _, in tqdm(db.test_loader):
            if args.cuda:
                article, question = article.cuda(), question.cuda()
            _p1, _p2 = model(article, question, is_training=False)  # N, T
            _p1 = F.softmax(_p1, dim=-1)
            _p2 = F.softmax(_p2, dim=-1)
            _p1 = _p1.cpu().data.numpy()
            _p2 = _p2.cpu().data.numpy()

            saved_p1.append(_p1)
            saved_p2.append(_p2)

            if model_id == 0:
                p1.append(_p1)
                p2.append(_p2)
            else:
                p1[batch_idx] = p1[batch_idx] + _p1
                p2[batch_idx] = p2[batch_idx] + _p2
            batch_idx += 1

        pickle.dump([saved_p1, saved_p2], open(model_path+"_result_"+str(args.max_article_len)+".pkl", "wb"))

    idx = 0
    for pp1, pp2 in zip(p1, p2):
        p1_predicted = np.argmax(pp1, axis=1).tolist()
        p2_predicted = np.argmax(pp2, axis=1).tolist()

        # _, p1_predicted = torch.max(pp1, 1)
        # _, p2_predicted = torch.max(pp2, 1)
        # p1_predicted = p1_predicted.cpu().numpy().tolist()
        # p2_predicted = p2_predicted.cpu().numpy().tolist()
        for _p1, _p2, _raw_article, _article_id, _question_id in zip(p1_predicted, p2_predicted,
                                                                     raw_article_list[idx:idx + len(p1_predicted)],
                                                                     article_id_list[idx:idx + len(p1_predicted)],
                                                                     question_id_list[idx:idx + len(p1_predicted)]):
            assert q2a[_question_id] == _article_id
            result[q2idx[_question_id]]["questions"].append({
                "questions_id": _question_id,
                "answer": "".join(_raw_article[_p1:_p2 + 1])
            })
        idx = idx + len(p1_predicted)

    print(len(result))
    json.dump(result, open(args.result_file, "w", encoding="utf-8"), ensure_ascii=False)

def gen_elmo_by_text(f, text, max_len):
    elmo_list = []
    count = 0
    dim = 1024
    for sent in text:
        sent = '\t'.join(sent)
        k = sent
        try:
            # tmp_len = len(f[k].value)
            # dim = f[k].value.shape[1]
            # zero_num = max_len - tmp_len
            one_elmo = f[k].value
            dim = one_elmo.shape[1]
            tmp_len = one_elmo.shape[0]
            zero_num = max_len - tmp_len
            if zero_num > 0:
                pad_value = np.zeros((zero_num, dim))
                # pad_value = np.repeat(np.array([0] * dim).reshape(1,-1), zero_num, axis=0)
                one_elmo = np.vstack([f[k].value, pad_value])
            elmo_list.append(one_elmo)
        except:
            #print(k)
            #print("error")
            #exit(1)
            one_elmo = np.zeros((max_len, dim))
            elmo_list.append(one_elmo)
            count += 1
    print("count:",count)
    elmo_list = np.array(elmo_list, dtype=np.float)
    return elmo_list
    # return torch.tensor(elmo_list, dtype=torch.float)

def gen_emlo_text(args):
    db = Data(args)
    db.load_vocab()
    db.build_test_dataset()

    test_raw_article_list = db.test_raw_article_list
    test_raw_question_list = db.test_raw_question_list

    question_hdf5_f = h5py.File(args.question_hdf5_path, "r")
    article_hdf5_f = h5py.File(args.article_hdf5_path, "r")

    test_idx = 0
    batch_id = 0

    for article, question, _, in tqdm(db.test_loader):
        tmp_test_raw_article_list = test_raw_article_list[test_idx:test_idx + question.size()[0]]
        tmp_test_raw_question_list = test_raw_question_list[test_idx:test_idx + question.size()[0]]
        question_elmo = gen_elmo_by_text(question_hdf5_f, tmp_test_raw_question_list, args.max_question_len)
        article_elmo = gen_elmo_by_text(article_hdf5_f, tmp_test_raw_article_list, args.max_article_len)
        elmo_save_path = "/backup231/lhliu/jszn/test_elmo/"+str(batch_id)+".pkl"
        pickle.dump((article_elmo, question_elmo), open(elmo_save_path, "wb"))
        print(elmo_save_path)
        test_idx += question.size()[0]
        batch_id += 1

def predict_elmo(args):
    db = Data(args)
    db.load_vocab()
    db.build_test_dataset()
    # model = BiDAF(args)
    model = SLQA(args)

    print("load model data:", args.model_path)
    model.load_state_dict(torch.load(args.model_path))
    if args.cuda:
        model = model.cuda()
    q2a = {}
    q2idx = {}
    result = []
    for i, d in enumerate(db.data):
        tmp = OrderedDict()
        tmp["article_id"] = d["article_id"]
        tmp["questions"] = []
        for q in d["questions"]:
            q2a[q["questions_id"]] = d["article_id"]
            q2idx[q["questions_id"]] = i
        result.append(tmp)
    raw_article_list = db.test_raw_article_list
    article_id_list = db.test_article_id_list
    question_id_list = db.test_question_id_list
    idx = 0
    batch_id = 0

    saved_p1 = []
    saved_p2 = []
    for article, question,  _, in tqdm(db.test_loader):
        if args.cuda:
            article, question = article.cuda(), question.cuda()
        elmo_save_path = "/backup231/lhliu/jszn/test_elmo/"+str(batch_id)+".pkl"
        article_elmo, question_elmo = pickle.load(open(elmo_save_path, "rb"))
        article_elmo = torch.tensor(article_elmo, dtype=torch.float)
        question_elmo = torch.tensor(question_elmo, dtype=torch.float)
        if args.cuda:
            question_elmo = question_elmo.cuda()
            article_elmo = article_elmo.cuda()

        batch_id += 1
        p1, p2 = model(article, question, article_elmo=article_elmo, question_elmo=question_elmo, is_training=False)

        saved_p1.append(F.softmax(p1, dim=-1).cpu().data.numpy())
        saved_p2.append(F.softmax(p2, dim=-1).cpu().data.numpy())

        _, p1_predicted = torch.max(p1, 1)
        _, p2_predicted = torch.max(p2, 1)
        p1_predicted = p1_predicted.cpu().numpy().tolist()
        p2_predicted = p2_predicted.cpu().numpy().tolist()
        for _p1, _p2, _raw_article, _article_id, _question_id in zip(p1_predicted, p2_predicted,
                                                                     raw_article_list[idx:idx+len(p1_predicted)],
                                                                     article_id_list[idx:idx+len(p1_predicted)],
                                                                     question_id_list[idx:idx + len(p1_predicted)]):
            assert q2a[_question_id] == _article_id
            result[q2idx[_question_id]]["questions"].append({
                "questions_id": _question_id,
                "answer": "".join(_raw_article[_p1:_p2 + 1])
            })
        idx = idx + len(p1_predicted)

    pickle.dump([saved_p1, saved_p2], open(args.model_path + "_result_" + str(args.max_article_len) + ".pkl", "wb"))

    print(len(result))
    json.dump(result, open(args.result_file, "w", encoding="utf-8"), ensure_ascii=False)

def predict_based_pkl(args):
    db = Data(args)
    db.load_vocab()
    db.build_test_dataset()

    model_path_list = args.model_path_list

    q2a = {}
    q2idx = {}
    result = []
    for i, d in enumerate(db.data):
        tmp = OrderedDict()
        tmp["article_id"] = d["article_id"]
        tmp["questions"] = []
        for q in d["questions"]:
            q2a[q["questions_id"]] = d["article_id"]
            q2idx[q["questions_id"]] = i
        result.append(tmp)
    raw_article_list = db.test_raw_article_list
    article_id_list = db.test_article_id_list
    question_id_list = db.test_question_id_list

    max_article_len = 2000

    pkl_list = []
    data_size = 0
    # model_path_list = ["./checkpoints/SLQA_epoch_12", "./checkpoints/SLQA2_epoch_7"]
    for model_id, model_path in enumerate(model_path_list):
        saved_p1, saved_p2 = pickle.load(open(model_path + "_result_" + str(max_article_len) + ".pkl", "rb"))
        saved_p1 = np.vstack(saved_p1)
        saved_p2 = np.vstack(saved_p2)
        data_size = saved_p2.shape[0]
        print(model_path, saved_p1.shape, saved_p2.shape)  # (100621, 2000)
        pkl_list.append((saved_p1, saved_p2))

    p1 = np.zeros((data_size, max_article_len))
    p2 = np.zeros((data_size, max_article_len))
    for i in range(data_size):

        for pkl in pkl_list:
            tmp_p1, tmp_p2 = pkl
            cur_p1 = tmp_p1[i]  # 2000
            cur_p2 = tmp_p2[i]
            b_idx = np.argmax(cur_p1)
            e_idx = np.argmax(cur_p2)
            # if b_idx <= e_idx:
            p1[i] += cur_p1
            p2[i] += cur_p2

    p1_predicted = np.argmax(p1, axis=1).tolist()  # (100621)
    p2_predicted = np.argmax(p2, axis=1).tolist()  # (100621)

    for _p1, _p2, _raw_article, _article_id, _question_id in zip(p1_predicted, p2_predicted,
                                                                 raw_article_list,
                                                                 article_id_list,
                                                                 question_id_list):
        assert q2a[_question_id] == _article_id
        result[q2idx[_question_id]]["questions"].append({
            "questions_id": _question_id,
            "answer": "".join(_raw_article[_p1:_p2 + 1])
        })

    print(len(result))
    json.dump(result, open(args.result_file, "w", encoding="utf-8"), ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action='store_true', default=False)
    parser.add_argument("--ensumble", action='store_true', default=False)

    parser.add_argument("--pre-embed", type=bool, default=False)
    parser.add_argument("--train-file", type=str, default="/disk/lhliu/szx_project/junshizhineng/train_data/chusai_preprocessed_seg.json")
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-path-list", type=list)

    parser.add_argument("--word2idx-path", type=str, default="./data/vocab/word2idx.json")
    parser.add_argument("--idx2word-path", type=str, default="./data/vocab/idx2word.json")
    parser.add_argument("--embed-size", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--drop-rate", type=float, default=0)

    # 下边的参数只有 predict() 才能用上
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--max-article-len", type=int, default=1200)  # 第二次提交结果是2000
    parser.add_argument("--max-question-len", type=int, default=142)
    parser.add_argument("--max-answer-len", type=int, default=13)  # not used

    parser.add_argument("--question-hdf5-path", type=str, default="/backup231/lhliu/jszn/test_question.hdf5")
    parser.add_argument("--article-hdf5-path", type=str, default="/backup231/lhliu/jszn/test_article.hdf5")

    args = parser.parse_args()
    print("dropout:", args.drop_rate)
    args.model_path_list = ["./checkpoints/SLQA_epoch_12", "./checkpoints/SLQA2_epoch_7",
                            "./checkpoints/SLQA_epoch_13", "./checkpoints/SLQA2_epoch_8",
                            "./checkpoints/SLQA3_epoch_12", "./checkpoints/SLQA3_epoch_13",
                            "./checkpoints/SLQA4_epoch_12", "./checkpoints/SLQA4_epoch_13",
                            "./checkpoints/SLQA_2000_1_epoch_11", "./checkpoints/SLQA_2000_1_epoch_12",
                            "./checkpoints/SLQA_2000_1_epoch_13", "./checkpoints/SLQA_2000_2_epoch_6",
                            "./checkpoints/SLQA_2000_2_epoch_7", "./checkpoints/SLQA5_epoch_11",
                            "./checkpoints/SLQA5_epoch_12", "./checkpoints/SLQA4_epoch_13"]
    if args.ensumble and args.model_path_list:
        predict_based_pkl(args)
    else:
        # predict_one_batch_size(args)
        print("single model predict ...")
        predict_elmo(args)
        # gen_emlo_text(args)
