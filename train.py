#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import argparse
from data.buil_dataset import Data
import datetime
from model.ema import EMA
from model.bidaf import BiDAF
from python.test_score import test_score
from logger import Logger

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

def train(args):
    db = Data(args)
    # db.build_vocab()  # 每次build_vocab，相同频数的字词id可能不同
    db.load_vocab()
    db.build_dataset()  # 得到train_loader

    model = BiDAF(args)
    if args.cuda:
        model = model.cuda()
    if args.ema:
        ema = EMA(0.999)
        print("Register EMA ...")
        for name, param in model.named_parameters():
           if param.requires_grad:
               ema.register(name, param.data)
    init_lr = args.init_lr
    optimizer = torch.optim.Adam(params=model.parameters(),  lr=init_lr)
    lr = init_lr

    batch_step = args.batch_step
    loss_fn = nn.CrossEntropyLoss()
    logger = Logger('./logs')
    step = 0

    valid_raw_article_list = db.valid_raw_article_list
    valid_answer_list = db.valid_answer_list

    print('========== Train ==============')

    for epoch in range(args.epoch_num):
        print('---Epoch', epoch, "lr:", lr)
        running_loss = 0.0
        count = 0
        print("len(db.train_loader):", len(db.train_loader))
        for article, question, answer_span, _ in db.train_loader:
            if args.cuda:
                article, question, answer_span = article.cuda(), question.cuda(), answer_span.cuda()
            p1, p2 = model(article, question)
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

                info = {'loss': running_loss / batch_step}
                running_loss = 0.0
                count = 0

                # 1. Log scalar values (scalar summary)
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step + 1)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), step + 1)
                    logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)
                step += 1

        # 验证集
        if args.with_valid:
            print('======== Epoch {} result ========'.format(epoch))
            print("len(db.valid_loader):", len(db.valid_loader))
            valid_result = []
            idx = 0
            for article, question, _ in db.valid_loader:
                if args.cuda:
                    article, question = article.cuda(), question.cuda()
                p1, p2 = model(article, question, is_trainning=False)

                _, p1_predicted = torch.max(p1.cpu().data, 1)
                _, p2_predicted = torch.max(p2.cpu().data, 1)
                p1_predicted = p1_predicted.numpy().tolist()
                p2_predicted = p2_predicted.numpy().tolist()
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

        lr = max(0.00001, init_lr * 0.9 ** (epoch + 1)) 
        print("lr:", lr)
        parameters = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=1e-7)

        # print(len(db.valid_loader))
        if epoch >= 1 and args.saved_model_file:
            torch.save(model.state_dict(), args.saved_model_file + "_epoch_" + str(epoch))
            print("saved model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-num", type=int, default=30)
    parser.add_argument("--cuda", action='store_true', default=False)
    parser.add_argument("--with-valid", action='store_true', default=False)
    parser.add_argument("--pre-embed", action='store_true', default=False)
    parser.add_argument("--ema", action='store_true', default=False)

    parser.add_argument("--max-article-len", type=int, default=1200)
    parser.add_argument("--max-question-len", type=int, default=142)
    parser.add_argument("--max-answer-len", type=int, default=13)  # not used
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--valid-batch-size", type=int, default=1)

    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--batch-step", type=int, default=300)

    parser.add_argument("--train-file", type=str, default="/disk/lhliu/szx_project/junshizhineng/train_data/question_with_span_seg.json")
    parser.add_argument("--saved-model-file", type=str)
    parser.add_argument("--word2idx-path", type=str, default="./data/vocab/word2idx.json")
    parser.add_argument("--idx2word-path", type=str, default="./data/vocab/idx2word.json")

    parser.add_argument("--init-lr", type=float, default=0.001)  # Adam: 0.001
    parser.add_argument("--embed-size", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--drop-rate", type=float, default=0.5)


    args = parser.parse_args()
    train(args)
