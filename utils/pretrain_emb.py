import pickle
import json
from gensim.models import Word2Vec
import numpy as np
import argparse
import logging
import os
import multiprocessing
import argparse
import json
from tqdm import tqdm

embed_size = 200

def save_dict(dic, fpath):
    output = open(fpath, 'wb')
    pickle.dump(dic, output)

def load_pickle(fpath):
    pkl_f = open(fpath, 'rb')
    return pickle.load(pkl_f)

def build_pretrain_emb(index2word, model_path, embedding_path):
    pretrain_emb = []
    model = Word2Vec.load(model_path)
    vocab_size = len(index2word)
    print("index2word len:", vocab_size)
    embedding_size = model.layer1_size
    print("embedding size:", embedding_size)
    count = 0
    for i in range(vocab_size):
        word = index2word[str(i)]
        if word in model.wv:
            pretrain_emb.append(model.wv[word])
        else:
            count += 1
            print("word random:", word)
            rand_emb = np.random.normal(loc=0, scale=1, size=embedding_size)
            pretrain_emb.append(rand_emb)
    print("rand init count:", count)
    pretrain_emb = np.array(pretrain_emb)
    np.save(embedding_path, pretrain_emb)

class MySentences(object):
    def __init__(self, fname_list):
        self.fname_list = fname_list

    def __iter__(self):
        for i, fname in enumerate(self.fname_list):
            for line in open(fname, 'r', encoding="utf-8"):
                sentences = json.loads(line, encoding="utf-8")
                yield sentences

def build_model(model_path, data_paths):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    sentences = MySentences(data_paths)  # a memory-friendly iterator
    model = Word2Vec(sentences, size=embed_size, window=5, min_count=5,
                     workers=multiprocessing.cpu_count(), sg=1)  # sg = 1: skip-gram

    model.save(model_path)

def preprocess_for_w2v(path_list):
    output_data = []
    for path in path_list:
        data = json.load(open(path, "r", encoding="utf-8"))
        for d in tqdm(data):
            article = d['article_content']
            output_data.append(json.dumps(article, ensure_ascii=False))
            questions = d["questions"]
            for q in questions:
                output_data.append(json.dumps(q['question'], ensure_ascii=False))
        print(len(output_data))
    with open("./w2v_saved_model/w2v_data.txt", "w", encoding="utf-8")as f:
        f.write("\n".join(output_data))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-path", type=str, default="../data/vocab/embed.200d.model")
    args = parser.parse_args()

    preprocess_for_w2v(["../../train_data/question_with_span_seg.json",
                        "../../train_data/chusai_preprocessed_seg.json"])
    input_file_list = ["./w2v_saved_model/w2v_data.txt"]
    model_path = "./w2v_saved_model/w2v.model"
    build_model(model_path, input_file_list)

    index2word = json.load(open("../data/vocab/idx2word.json", "r", encoding="utf-8"))
    build_pretrain_emb(index2word, amodel_path, args.embedding_path)
    print("save in", args.embedding_path)