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

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = F.softmax

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output

class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''

    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.w_qs)
        init.xavier_normal_(self.w_ks)
        init.xavier_normal_(self.w_vs)

    def forward(self, q, k, v, attn_mask):
        # q: N, T, 2d
        # k: N, T, 2d
        # v: N, T, 2d
        # attn_mask: N T T

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)  # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)  # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)  # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        outputs = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)

        # project back to residual size, 相加而已
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual)

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)  # position-wise
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)

def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.eq(Constants.PAD).unsqueeze(1)  # N, 1, T
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)  # N T T
    return pad_attn_mask

class BiDAF(nn.Module):
    def __init__(self, args):
        super(BiDAF, self).__init__()
        self.embed_size = args.embed_size
        self.d = args.hidden_size
        self.args = args
        self.dropout = nn.Dropout(p=self.args.drop_rate)
        self.word_embedding = nn.Embedding(args.word_vocab_size, args.embed_size, padding_idx=Constants.PAD)

        if args.pre_embed:
            print("loading pretrain embedding ...")
            # self.embedding.weight = nn.Parameter(args.pre_embd_w, requires_grad=is_train_embd)
            self.word_embedding.weight.data.copy_(torch.from_numpy(np.load(embedding_path)))


        self.article_context_embedding = nn.LSTM(self.embed_size, self.d, bidirectional=True, batch_first=True)
        self.question_context_embedding = nn.LSTM(self.embed_size, self.d, bidirectional=True, batch_first=True)

        # self.W = nn.Linear(6*self.d, 1, bias=False)

        self.modeling_layer = nn.LSTM(8*self.d, self.d, bidirectional=True, batch_first=True)

        self.p1_layer = nn.Linear(10*self.d, 1)
        self.p2_lstm_layer = nn.LSTM(2*self.d, self.d, bidirectional=True, batch_first=True)
        self.p2_layer = nn.Linear(10*self.d, 1)

        # self.slf_attn = MultiHeadAttention(
        #      n_head=6, d_model=2*self.d, d_k=64, d_v=64, dropout=dropout)
        # self.pos_ffn = PositionwiseFeedForward(d_hid=2*self.d, d_inner_hid=100, dropout=dropout)
        self.T = 2
        self.c2q_modeling_layer = nn.LSTM(2*self.d, self.d, bidirectional=True, batch_first=True)


    def forward(self, article, question, is_trainning=True):
        # batch_size, len
        if is_trainning:
            self.dropout = nn.Dropout(p=self.args.drop_rate)
        else:
            # 验证的时候是0
            self.dropout = nn.Dropout(p=0)

        batch_size = article.size(0)
        T = article.size(1)   # context sentence length (word level)
        J = question.size(1)  # query sentence length   (word level)

        # mask
        article_mask = article.eq(Constants.PAD)
        _article_mask = article_mask.unsqueeze(1).expand(batch_size, T, T)
        question_mask = question.eq(Constants.PAD)
        _question_mask = question_mask.unsqueeze(1).expand(batch_size, T, J)

        word_embd = self.word_embedding(article)  # (N, T, embd_size)
        embd_context, _ = self.article_context_embedding(word_embd)  # (N, T, 2d)
        embd_context = self.dropout(embd_context)

        question_embd = self.word_embedding(question)
        embd_query, _ = self.question_context_embedding(question_embd)  # (N, J, 2d)
        embd_query = self.dropout(embd_query)

        # 对embd_context, question_embd 做 self-attention
        # embd_context = self.slf_attn(
        #     embd_context, embd_context, embd_context, attn_mask=get_attn_padding_mask(article, article))  # N T T
        # embd_context = self.pos_ffn(embd_context)  # 先不stack，transformer stack了6次
        #
        # embd_query = self.slf_attn(
        #     embd_query, embd_query, embd_query, attn_mask=get_attn_padding_mask(question, question))  # N T T
        # embd_query = self.pos_ffn(embd_query)  # 先不stack，transformer stack了6次

        # 4. 多轮 Attention Flow Layer

        # shape = (batch_size, T, J, 2 * self.d)  # (N, T, J, 2d)
        # embd_context_ex = embd_context.unsqueeze(2)  # (N, T, 1, 2d)
        # embd_context_ex = embd_context_ex.expand(shape)  # (N, T, J, 2d)
        # embd_query_ex = embd_query.unsqueeze(1)         # (N, 1, J, 2d)
        # embd_query_ex = embd_query_ex.expand(shape)  # (N, T, J, 2d)
        # a_elmwise_mul_b = torch.mul(embd_context_ex, embd_query_ex)  # (N, T, J, 2d)
        # cat_data = torch.cat((embd_context_ex, embd_query_ex, a_elmwise_mul_b), 3)  # (N, T, J, 6d), [h;u;h◦u]
        # S = self.W(cat_data).view(batch_size, T, J)  # (N, T, J)

        # Context2Query, context中每个词，对query中每个词的权重，也就是对query进行softmax
        # 可以加self-attention
        # 应该对J中每个词进行mask。对谁做softmax就对谁做mask。
        for i in range(self.T):
            if i == 0:
                S = torch.bmm(embd_context, embd_query.transpose(1, 2))   # (N, T, J)
                S.masked_fill_(_question_mask, -float('inf'))  # Todo 测试
                c2q = torch.bmm(F.softmax(S, dim=-1), embd_query)  # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )
                c2q, _ = self.c2q_modeling_layer(c2q)
                c2q = self.dropout(c2q)

            else:
                S = torch.bmm(c2q, embd_query.transpose(1, 2))   # (N, T, J)
                S.masked_fill_(_question_mask, -float('inf'))  # Todo 测试
                c2q = torch.bmm(F.softmax(S, dim=-1), embd_query)  # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )
                c2q, _ = self.c2q_modeling_layer(c2q)
                c2q = self.dropout(c2q)

        # Query2Context，
        # context中每个词与整体query相似度最大的单词
        S = torch.bmm(c2q, embd_query.transpose(1, 2))  # (N, T, J)
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

        # fuse 后做 self-attention
        slf_att = torch.bmm(M, M.transpose(1, 2))  # N T T
        slf_att.masked_fill_(_article_mask, -float('inf'))  # Todo 测试
        M = torch.bmm(F.softmax(slf_att, dim=-1), M)  # N, T, 2d

        # 6. Output Layer
        # TODO 进行特殊设计，现在的做法其实也可以根据start预测end
        G_M = torch.cat((G, M), 2)  # (N, T, 10d)
        p1 = self.p1_layer(G_M).squeeze(dim=2)  # (N, T)

        M2, _ = self.p2_lstm_layer(M)  # (N, T, 2d)
        M2 = self.dropout(M2)
        # 再做self—attention，再得到p1的情况下，找p2
        slf_att = torch.bmm(M2, M2.transpose(1, 2))  # N T T
        slf_att.masked_fill_(_article_mask, -float('inf'))  # Todo 测试
        M2 = torch.bmm(F.softmax(slf_att, dim=-1), M2)  # N, T, 2d

        G_M2 = torch.cat((G, M2), 2)  # (N, T, 10d)
        p2 = self.p2_layer(G_M2).squeeze(dim=2)  # (N, T)

        return p1, p2
