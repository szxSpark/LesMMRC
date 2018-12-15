import torch
import torch.nn as nn
import torch.nn.functional as F
from model.highway import Highway
import numpy as np
import torch.nn.init as init

embedding_path = "/disk/lhliu/szx_project/junshizhineng/BIDAF/data/vocab/embed.100d.model.npy"

class ConstantsClass():
    def __init__(self):
        self.PAD = 1

Constants = ConstantsClass()
dropout = 0

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

class BiDAF(nn.Module):
    def __init__(self, args):
        super(BiDAF, self).__init__()
        self.embed_size = args.embed_size
        self.d = args.hidden_size
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.word_embedding = nn.Embedding(args.word_vocab_size, args.embed_size, padding_idx=Constants.PAD)

        if args.pre_embed:
            print("loading pretrain embedding ...")
            # self.embedding.weight = nn.Parameter(args.pre_embd_w, requires_grad=is_train_embd)
            self.word_embedding.weight.data.copy_(torch.from_numpy(np.load(embedding_path)))


        self.context_embedding = nn.GRU(self.embed_size, self.d, bidirectional=True, batch_first=True)

        self.W = nn.Linear(6*self.d, 1, bias=False)

        self.modeling_layer = nn.GRU(8*self.d, self.d, bidirectional=True, batch_first=True, num_layers=2, dropout=args.drop_rate)
        self.context_self_attn = nn.Linear(2*self.d, 2*self.d, bias=False)
        self.self_attn_fuse = Fuse(2*self.d, 2*self.d, num=2)
        self.comp_self_attn = nn.GRU(2*self.d, self.d, bidirectional=True, batch_first=True, num_layers=2, dropout=args.drop_rate)
        self.p1_layer = nn.Linear(10*self.d, 1)
        self.p2_lstm_layer = nn.GRU(2*self.d, self.d, bidirectional=True, batch_first=True, num_layers=2, dropout=args.drop_rate)
        self.p2_layer = nn.Linear(10*self.d, 1)

        # self.slf_attn = MultiHeadAttention(
        #      n_head=6, d_model=2*self.d, d_k=64, d_v=64, dropout=dropout)
        # self.pos_ffn = PositionwiseFeedForward(d_hid=2*self.d, d_inner_hid=100, dropout=dropout)


    def forward(self, article, question):
        # batch_size, len
        batch_size = article.size(0)
        T = article.size(1)   # context sentence length (word level)
        J = question.size(1) # query sentence length   (word level)

        # TODO word_dropout

        # mask
        article_mask = article.eq(Constants.PAD)
        question_mask = question.eq(Constants.PAD)

        word_embd = self.word_embedding(article)  # (N, T, embd_size)
        embd_context, _ = self.context_embedding(word_embd)  # (N, T, 2d)
        embd_context = self.dropout(embd_context)

        question_embd = self.word_embedding(question)
        embd_query, _ = self.context_embedding(question_embd)  # (N, J, 2d)
        embd_query = self.dropout(embd_query)

        shape = (batch_size, T, J, 2 * self.d)  # (N, T, J, 2d)
        embd_context_ex = embd_context.unsqueeze(2)  # (N, T, 1, 2d)
        embd_context_ex = embd_context_ex.expand(shape)  # (N, T, J, 2d)
        embd_query_ex = embd_query.unsqueeze(1)         # (N, 1, J, 2d)
        embd_query_ex = embd_query_ex.expand(shape)  # (N, T, J, 2d)
        a_elmwise_mul_b = torch.mul(embd_context_ex, embd_query_ex)  # (N, T, J, 2d)
        cat_data = torch.cat((embd_context_ex, embd_query_ex, a_elmwise_mul_b), 3)  # (N, T, J, 6d), [h;u;h◦u]
        S = self.W(cat_data).view(batch_size, T, J)  # (N, T, J)

        # Context2Query, context中每个词，对query中每个词的权重，也就是对query进行softmax
        # 可以加self-attention
        # 应该对J中每个词进行mask。对谁做softmax就对谁做mask。
        _question_mask = question_mask.unsqueeze(1).expand(batch_size, T, J)
        S.masked_fill_(_question_mask, -float('inf'))  # Todo 测试
        c2q = torch.bmm(F.softmax(S, dim=-1), embd_query)  # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )

        # Query2Context，
        # context中每个词与整体query相似度最大的单词
        max_tmp = torch.max(S, 2)[0]   # (N, T)
        max_tmp.masked_fill_(article_mask, -float('inf'))  # Todo 测试
        b = F.softmax(max_tmp, dim=-1)  # (N, T)
        q2c = torch.bmm(b.unsqueeze(1), embd_context)  # (N, 1, 2d) = bmm( (N, 1, T), (N, T, 2d) )
        q2c = q2c.repeat(1, T, 1)  # (N, T, 2d), tiled T times

        # G: query aware representation of each context word
        G = torch.cat((embd_context, c2q, embd_context.mul(c2q), embd_context.mul(q2c)), 2)  # (N, T, 8d)

        # 5. Modeling Layer
        M, _ = self.modeling_layer(G)  # M: (N, T, 2d)
        M = self.dropout(M)
        _article_mask = article_mask.unsqueeze(1).expand(batch_size, T, T)
        self_score = torch.bmm(self.context_self_attn(M), M.transpose(1,2))
        self_score.masked_fill_(_article_mask, -float('inf'))
        M_hat = torch.bmm(F.softmax(self_score, dim=-1), M)
        M_fuse = self.self_attn_fuse(M, M_hat)
        M, _ = self.comp_self_attn(M_fuse)
        # 6. Output Layer
        # TODO 进行特殊设计，现在的做法其实也可以根据start预测end
        G_M = torch.cat((G, M), 2)  # (N, T, 10d)
        p1 = self.p1_layer(G_M).squeeze(dim=2)  # (N, T)

        M2, _ = self.p2_lstm_layer(M)  # (N, T, 2d)
        M2 = self.dropout(M2)

        G_M2 = torch.cat((G, M2), 2)  # (N, T, 10d)
        p2 = self.p2_layer(G_M2).squeeze(dim=2)  # (N, T)

        return p1, p2
