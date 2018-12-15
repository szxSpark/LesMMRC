#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import argparse
#from data.buil_dataset import Data
from data.llh_build_dataset import Data
import datetime
from model.ema import EMA
from model.llh_bidaf import BiDAF
from model.llh_slqa2 import SLQA
from python.test_score import test_score
from logger import Logger
import numpy as np
import h5py
import pickle


# def custom_loss_fn(data, labels):
#     loss = torch.zeros(1)
#     print(data.size(), labels.size())
#     for d, label in zip(data, labels):
#         print(d.size(), label.size())
#         try:
#             loss -= torch.log(d[label]).cpu()
#         except Exception as e:
#             print(d)
#             print(label)
#             print(e)
#     loss /= data.size(0)
#     return loss

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
    #print("count:",count)
    # elmo_list = np.array(elmo_list, dtype=np.float)
    # return elmo_list
    return torch.tensor(elmo_list, dtype=torch.float)

def train(args):
    db = Data(args)
    # db.build_vocab()  # 每次build_vocab，相同频数的字词id可能不同
    db.load_vocab()
    db.build_dataset()  # 得到train_loader

    # model = BiDAF(args)
    model = SLQA(args)
    if args.cuda:
        model = model.cuda()
    if args.ema:
        ema = EMA(0.999)
        print("Register EMA ...")
        for name, param in model.named_parameters():
           if param.requires_grad:
               ema.register(name, param.data)
    init_lr = args.init_lr
    parameters = filter(lambda param: param.requires_grad, model.parameters())
    weight_decay=1e-6
    #weight_decay=0
    optimizer = torch.optim.Adam(params=parameters,  lr=init_lr, weight_decay=weight_decay)
    batch_step = args.batch_step
    loss_fn = nn.CrossEntropyLoss()
    logger = Logger('./logs')
    step = 0

    train_raw_article_list = db.train_raw_article_list
    train_raw_question_list = db.train_raw_question_list

    valid_raw_article_list = db.valid_raw_article_list
    valid_answer_list = db.valid_answer_list
    valid_raw_question_list = db.valid_raw_question_list

    #question_hdf5_f = h5py.File(args.question_hdf5_path, "r")
    # article_hdf5_f = h5py.File(args.article_hdf5_path, "r")
    print('========== Train ==============')
    for epoch in range(args.epoch_num):
        print('---Epoch', epoch)
        running_loss = 0.0
        count = 0
        print("len(db.train_loader):", len(db.train_loader))
        train_idx = 0
        for batch_id, (article, question, answer_span, _) in enumerate(db.train_loader):
            if args.cuda:
                article, question, answer_span = article.cuda(), question.cuda(), answer_span.cuda()
            #tmp_train_raw_article_list = train_raw_article_list[train_idx:train_idx + question.size()[0]]
            #tmp_train_raw_question_list = train_raw_question_list[train_idx:train_idx + question.size()[0]]
            #question_elmo = gen_elmo_by_text(question_hdf5_f, tmp_train_raw_question_list, args.max_question_len)
            # article_elmo = gen_elmo_by_text(article_hdf5_f, tmp_train_raw_article_list, args.max_article_len)
            # elmo_save_path = "/backup231/lhliu/jszn/elmo/"+str(batch_id)+".pkl"
            # pickle.dump((article_elmo, question_elmo), open(elmo_save_path, "wb"))
            # print(elmo_save_path)
            train_idx += question.size()[0]
            # continue
            #if args.cuda:
            #    question_elmo = question_elmo.cuda()
                # article_elmo = article_elmo.cuda()
            
            # continue

            p1, p2 = model(article, question, T_step=3)
            loss_p1 = loss_fn(p1, answer_span.transpose(0,1)[0])
            loss_p2 = loss_fn(p2, answer_span.transpose(0,1)[1])
            running_loss += loss_p1.item()
            running_loss += loss_p2.item()

            optimizer.zero_grad()
            (loss_p1+loss_p2).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            if args.ema:
               for name, param in model.named_parameters():
                   if param.requires_grad:
                       param.data = ema(name, param.data)

            count += 1
            if count % batch_step == 0:
                rep_str = '[{}] Epoch {}, loss: {:.3f}'
                print(rep_str.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                                     epoch,
                                     running_loss / batch_step))

                # info = {'loss': running_loss / batch_step}
                running_loss = 0.0
                count = 0

                # # 1. Log scalar values (scalar summary)
                # for tag, value in info.items():
                #     logger.scalar_summary(tag, value, step + 1)

                # # 2. Log values and gradients of the parameters (histogram summary)
                # for tag, value in model.named_parameters():
                #     tag = tag.replace('.', '/')
                #     logger.histo_summary(tag, value.data.cpu().numpy(), step + 1)
                #     logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)
                step += 1
          #  break
        # 验证集
        if args.with_valid:
            print('======== Epoch {} result ========'.format(epoch))
            print("len(db.valid_loader):", len(db.valid_loader))
            valid_result = []
            idx = 0
            for article, question, _ in db.valid_loader:
                if args.cuda:
                    article, question = article.cuda(), question.cuda()

                #tmp_valid_raw_article_list = valid_raw_article_list[idx:idx+question.size()[0]]
                #tmp_valid_raw_question_list = valid_raw_question_list[idx:idx+question.size()[0]]
                #question_elmo = gen_elmo_by_text(question_hdf5_f, tmp_valid_raw_question_list, args.max_question_len)
                #article_elmo = gen_elmo_by_text(article_hdf5_f, tmp_valid_raw_article_list, args.max_article_len)
                #if args.cuda:
                #    question_elmo = question_elmo.cuda()
                #    article_elmo = article_elmo.cuda()
                p1, p2 = model(article, question, is_training=False, T_step=3)

                _, p1_predicted = torch.max(p1.cpu().data, 1)
                _, p2_predicted = torch.max(p2.cpu().data, 1)
                p1_predicted = p1_predicted.numpy().tolist()
                p2_predicted = p2_predicted.numpy().tolist()
                assert question.size()[0] == len(p1_predicted)
                for _p1, _p2, _raw_article, _answer in zip(p1_predicted, p2_predicted,
                                                           valid_raw_article_list[idx:idx+len(p1_predicted)],
                                                           valid_answer_list[idx:idx+len(p1_predicted)]):
                    valid_result.append({
                        "ref_answer": _answer,
                        "cand_answer": "".join(_raw_article[_p1:_p2 + 1])
                    })
                idx = idx+len(p1_predicted)
            rouge_score = test_score(valid_result)
            info = {'rouge_score': rouge_score}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch + 1)
        #lr = init_lr
        lr = max(0.00001, init_lr * 0.9 ** (epoch + 1))  # 考虑是否使用
        print("lr:", lr)
        parameters = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params=parameters,  lr=lr, weight_decay=weight_decay)

        # print(len(db.valid_loader))
        if epoch >= 0 and args.saved_model_file:
            torch.save(model.state_dict(), args.saved_model_file + "_epoch_" + str(epoch))
            print("saved model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-num", type=int, default=20)
    parser.add_argument("--cuda", action='store_true', default=False)
    parser.add_argument("--with-valid", action='store_true', default=False)
    parser.add_argument("--pre-embed", action='store_true', default=False)
    parser.add_argument("--ema", action='store_true', default=False)


    parser.add_argument("--max-article-len", type=int, default=1200)
    parser.add_argument("--max-question-len", type=int, default=142)
    parser.add_argument("--max-answer-len", type=int, default=13)  # not used
    parser.add_argument("--batch-size", type=int, default=64) # don't change
    parser.add_argument("--valid-batch-size", type=int, default=32)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--batch-step", type=int, default=500)

    parser.add_argument("--train-file", type=str, default="/disk/lhliu/szx_project/junshizhineng/train_data/question_with_span_seg.json")
    parser.add_argument("--saved-model-file", type=str)
    parser.add_argument("--model-id", type=int, default=9)
    parser.add_argument("--word2idx-path", type=str, default="./data/vocab/word2idx.json")
    parser.add_argument("--idx2word-path", type=str, default="./data/vocab/idx2word.json")

    parser.add_argument("--init-lr", type=float, default=0.001)  # Adam: 0.001
    parser.add_argument("--embed-size", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--drop-rate", type=float, default=0.5)

    parser.add_argument("--question-hdf5-path", type=str, default="/disk/lhliu/szx_project/junshizhineng/ELMoForManyLangs/question.hdf5")
    parser.add_argument("--article-hdf5-path", type=str, default="/disk/lhliu/szx_project/junshizhineng/ELMoForManyLangs/article.hdf5")

    args = parser.parse_args()
    train(args)
