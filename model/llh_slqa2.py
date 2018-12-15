import torch
import torch.nn as nn
import torch.nn.functional as F
from model.highway import Highway
import numpy as np
import torch.nn.init as init

embedding_path = "/disk/lhliu/szx_project/junshizhineng/BIDAF/data/vocab/embed.200d.model.npy"

class ConstantsClass():
    def __init__(self):
        self.PAD = 1

Constants = ConstantsClass()
dropout = 0

class Attn(nn.Module):
    def __init__(self, input_size, attn_size):
        super().__init__()
        self.W_lin = nn.Linear(input_size, attn_size, bias=False)
    
    def forward(self, p, q):
        # key is passage, query is query
        p_out = F.relu(self.W_lin(p)) # N*p_len*attn_size
        q_out = F.relu(self.W_lin(q)) # N*q_len*attn_size
        score = torch.bmm(p_out, q_out.transpose(1,2))
        return score

class Fuse(nn.Module):
    def __init__(self, input_size, output_size, num=2):
        super().__init__()
        self.input_size = input_size
        self.num = num
        self.fuse_linear = nn.Linear(input_size*(num+2*num-2), output_size)
        self.gate_linear = nn.Linear(input_size*(num+2*num-2), 1)

    def forward(self, input1, input2, input3=None):
        if self.num == 3:
            x = torch.cat([input1, input2, input3, input1.mul(input2), input1-input2,
                        input1.mul(input3), input1-input3], dim=2)
        if self.num == 2:
            x = torch.cat([input1, input2, input1.mul(input2), input1-input2], dim=2)
        m = F.tanh(self.fuse_linear(x))
        g = F.sigmoid(self.gate_linear(x))
        output = g * m + (1-g)*input1
        return output

class SLQA(nn.Module):
    def __init__(self, args):
        super(SLQA, self).__init__()
        self.args = args
        self.embed_size = args.embed_size
        self.d = args.hidden_size
        self.drop_rate = args.drop_rate
        self.word_embedding = nn.Embedding(args.word_vocab_size, args.embed_size, padding_idx=Constants.PAD)

        if args.pre_embed:
            print("loading pretrain embedding ...")
            # self.embedding.weight = nn.Parameter(args.pre_embd_w, requires_grad=is_train_embd)
            self.word_embedding.weight.data.copy_(torch.from_numpy(np.load(embedding_path)))
            self.word_embedding.weight.requires_grad = False
        self.elmo_dim = 0
        self.context_embedding = nn.LSTM(self.embed_size+self.elmo_dim, self.d, bidirectional=True, batch_first=True)
        # self.elmo_dim = 1024
        self.query_embedding = nn.LSTM(self.embed_size+self.elmo_dim, self.d, bidirectional=True, batch_first=True)
        self.p_q_attn = Attn(input_size=2*self.d, attn_size=self.d)
        self.p_fuse = Fuse(2*self.d, 2*self.d, num=2)
        self.q_fuse = Fuse(2*self.d, 2*self.d, num=2)
        self.W = nn.Linear(6*self.d, 1, bias=False)
        self.comp_p_rnn = nn.LSTM(2*self.d, self.d, bidirectional=True, batch_first=True, num_layers=1, dropout=args.drop_rate)
        self.comp_q_rnn = nn.LSTM(2*self.d, self.d, bidirectional=True, batch_first=True, num_layers=1, dropout=args.drop_rate)

        self.context_self_attn = nn.Linear(2*self.d, 2*self.d, bias=False)
        self.self_attn_fuse = Fuse(2*self.d, 2*self.d, num=2)
        self.comp_self_attn = nn.LSTM(2*self.d, self.d, bidirectional=True, batch_first=True, num_layers=1, dropout=args.drop_rate)
        self.self_q_attn = nn.Linear(self.d*2, 1, bias=False)
        
        self.W_s = nn.Linear(2*self.d, 2*self.d, bias=False)
        self.W_e = nn.Linear(2*self.d, 2*self.d, bias=False)
        

    def forward(self, article, question, article_elmo=None, question_elmo=None, is_training=True, T_step=0):
        if not is_training:
            self.drop_rate = 0.0
        else:
            self.drop_rate = self.args.drop_rate
        # batch_size, len
        batch_size = article.size(0)
        T = article.size(1)   # context sentence length (word level)
        J = question.size(1) # query sentence length   (word level)

        # TODO word_dropout

        # mask
        article_mask = article.eq(Constants.PAD)
        question_mask = question.eq(Constants.PAD)

        word_embd = self.word_embedding(article)  # (N, T, embd_size)
        if article_elmo is not None:
            word_embd = torch.cat([word_embd, article_elmo], 2)
        word_embd = F.dropout(word_embd, p=self.drop_rate, training=is_training)
        embd_context, _ = self.context_embedding(word_embd)  # (N, T, 2d)
        embd_context = F.dropout(embd_context, p=self.drop_rate, training=is_training)
        
        question_embd = self.word_embedding(question)
        if question_elmo is not None:
            question_embd = torch.cat([question_embd, question_elmo], 2)
        question_embd = F.dropout(question_embd, p=self.drop_rate, training=is_training)
        embd_query, _ = self.query_embedding(question_embd)  # (N, J, 2d)
        embd_query = F.dropout(embd_query, p=self.drop_rate, training=is_training)
        # S = torch.bmm(embd_context, embd_query.transpose(1, 2))

        # Context2Query, context中每个词，对query中每个词的权重，也就是对query进行softmax
        # 可以加self-attention
        # 应该对J中每个词进行mask。对谁做softmax就对谁做mask。
        
        S = self.p_q_attn(embd_context, embd_query)
        S_copy = torch.rand(batch_size, T, J).cuda()
        S_copy.copy_(S)
        _question_mask = question_mask.unsqueeze(1).expand(batch_size, T, J)
        S.masked_fill_(_question_mask, -float('inf'))  # Todo 测试
        c2q = torch.bmm(F.softmax(S, dim=-1), embd_query)  # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )
        P = self.p_fuse(embd_context, c2q)
        P = F.dropout(P, p=self.drop_rate, training=is_training)
        P, _ = self.comp_p_rnn(P)
        P = F.dropout(P, p=self.drop_rate, training=is_training)

        # Query2Context，
        _article_mask = article_mask.unsqueeze(2).expand(batch_size, T, J)
        S_copy.masked_fill_(_article_mask, -float('inf')) 
        q2c = torch.bmm(F.softmax(S_copy.transpose(1,2), dim=-1), embd_context)  # (N, T, 2d)
        Q = self.q_fuse(embd_query, q2c)
        Q = F.dropout(Q , p=self.drop_rate, training=is_training)
        Q, _ = self.comp_q_rnn(Q)
        Q = F.dropout(Q , p=self.drop_rate, training=is_training)
        # self p attn
        _article_mask = article_mask.unsqueeze(1).expand(batch_size, T, T)
        self_score = torch.bmm(self.context_self_attn(P), P.transpose(1,2))
        self_score.masked_fill_(_article_mask, -float('inf'))
        P_hat = torch.bmm(F.softmax(self_score, dim=-1), P)
        P_fuse = self.self_attn_fuse(P, P_hat)
        P_fuse = F.dropout(P_fuse , p=self.drop_rate, training=is_training)
        P, _ = self.comp_self_attn(P_fuse) # N*T*2d
        P = F.dropout(P, p=self.drop_rate, training=is_training)
        # self q attn
        self_q_attn_score = self.self_q_attn(Q) # b*q_len*1
        # self_q_attn_score.masked_fill_(question_mask.unsqueeze(2), -float('inf'))  # Todo 测试
        # print(self_q_attn_score)
        q_v = torch.bmm(F.softmax(self_q_attn_score.transpose(1,2), dim=2), Q) # b*1*dim
        # print(q_v)
        # 6. Output Layer
        p1 = torch.bmm(self.W_s(q_v), P.transpose(1,2)).squeeze(1)
        p2 = torch.bmm(self.W_e(q_v), P.transpose(1,2)).squeeze(1)

        return p1, p2
