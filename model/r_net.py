import torch
from torch import nn
import torch.nn.functional as F
from modules.attention import AttentionPooling
from modules.recurrent import StackedCell, AttentionEncoderCell, AttentionEncoder


dropout_rate = 0.2
attn_size = 75
attn_mode = 'mlp'
hidden_size = 75

class SentenceEncoding(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
            bidirectional, dropout):
        super().__init__()

        self.question_encoder = nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bidirectional=bidirectional,
                                dropout=dropout,
                                batch_first=True)

        self.passage_encoder = nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bidirectional=bidirectional,
                                dropout=dropout,
                                batch_first=True)

    def forward(self, question_embedding, passage_embedding):
        # [batch_size, question_len, embedding_dim] ->
        # [batch_size, question_len, hidden_size*num_layers]
        question_outputs, _ = self.question_encoder(question_embedding)

        # [batch_size, passage_len, embedding_dim] ->
        # [batch_size, passage_len, hidden_size*num_layers]
        passage_outputs, _ = self.passage_encoder(passage_embedding)

        return question_outputs, passage_outputs


class PairEncoder(nn.Module):
    def __init__(self, question_embedding_dim, passage_embedding_dim,
                cell_factory=AttentionEncoderCell):
        super().__init__()
        attn_args = [question_embedding_dim, passage_embedding_dim, hidden_size]
        attn_kwags = {'attn_size': attn_size, 'batch_first': True, 'mode': attn_mode}

        self.pair_encoder = AttentionEncoder(
                cell_factory, question_embedding_dim, passage_embedding_dim,
                hidden_size, AttentionPooling, attn_args, attn_kwags,
                bidirectional=False, mode='GRU',
                num_layers=1, dropout=dropout_rate,
                residual=False, gated=True,
                rnn_cell=nn.GRUCell, attn_mode="pair_encoding"
        )

    def forward(self, questions, passage):
        inputs = (passage, questions)
        result = self.pair_encoder(inputs)
        return result


class SelfMatchingEncoder(nn.Module):

    def __init__(self, passage_embedding_dim,
                cell_factory=AttentionEncoderCell):
        super().__init__()
        attn_args = [passage_embedding_dim, passage_embedding_dim]
        attn_kwags = {'attn_size': attn_size, 'batch_first': True, 'mode': attn_mode}

        self.self_match_encoder = AttentionEncoder(
                cell_factory, passage_embedding_dim, passage_embedding_dim,
                hidden_size, AttentionPooling, attn_args, attn_kwags,
                bidirectional=True, mode='GRU',
                num_layers=1, dropout=dropout_rate,
                residual=False, gated=True,
                rnn_cell=nn.GRUCell, attn_mode="self_matching"
        )

    def forward(self, passage):
        inputs = (passage, passage)
        result = self.self_match_encoder(inputs)
        return result


class PointerNetwork(nn.Module):
    def __init__(self, question_size, passage_size, hidden_size, attn_size=None,
                 cell_type=nn.GRUCell, num_layers=1, dropout=0, residual=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        if attn_size is None:
            attn_size = question_size

        # TODO: what is V_q? (section 3.4)
        v_q_size = question_size
        self.question_pooling = AttentionPooling(question_size,
                                                 v_q_size, attn_size=attn_size, mode=attn_mode)
        self.passage_pooling = AttentionPooling(passage_size,
                                                question_size, attn_size=attn_size, mode=attn_mode)
        # using xavier_initializer: https://github.com/minsangkim142/R-net/blob/master/layers.py#L151
        self.V_q = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, 1, v_q_size)), requires_grad=True)
        self.cell = StackedCell(question_size, question_size, num_layers=num_layers,
                                dropout=dropout, rnn_cell=cell_type, residual=residual, **kwargs)

    def forward(self, question, passage):
        hidden = self.question_pooling(question, self.V_q,
                                       broadcast_key=True)  # batch x 1 x n
        # print("point question hidden size:", hidden.size())
        hidden = hidden.transpose(0, 1)  # 1 * batch * n
        # num_layers * batch * n
        hidden = hidden.expand([self.num_layers, hidden.size(1), hidden.size(2)])

        hidden = hidden.transpose(0, 1)  # batch * 1 * n
        # print("--------PointerNetwork passage pooling--------")
        inputs, ans_begin = self.passage_pooling(passage, hidden,
                                                 key_mask=None, return_key_scores=True)
        # print("ans_begin size:", ans_begin.size())  # batch_size * passage_len
        _, hidden = self.cell(inputs.squeeze(1), hidden.transpose(0, 1)) # 1 * batch_size * n
        _, ans_end = self.passage_pooling(passage, hidden.transpose(0, 1),
                                          key_mask=None, return_key_scores=True)

        return ans_begin, ans_end


class RNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.word_embedding = nn.Embedding(args.word_vocab_size, args.embed_size)
        # if args.pre_embed:
        #     print("loading pretrain embedding ...")
        #     self.embedding.weight = nn.Parameter(args.pre_embd_w, requires_grad=is_train_embd)

        self.sentence_encoding = SentenceEncoding(args.embed_size,
                hidden_size, num_layers=3, bidirectional=True, dropout=dropout_rate)

        sentence_encoding_dim = hidden_size * 2     # bidirectional = True
        self.pair_encoder = PairEncoder(sentence_encoding_dim, sentence_encoding_dim)

        pair_encoding_dim = hidden_size * 1     # bidirectional = False

        self.self_matching_encoder = SelfMatchingEncoder(pair_encoding_dim)

        question_dim = hidden_size * 2  # sentence encoding bidirectional is True
        passage_dim = hidden_size * 2  # self matching encoding bidirectional is True
        self.pointer_output = PointerNetwork(question_dim, passage_dim, hidden_size,
                                            dropout=dropout_rate)


    def forward(self, question, passage):
        # In: batch_size * question_len
        # Out: batch_size * question_len * word_embedding_dim
        question_embedding = self.word_embedding(question)
        # print("question embedding size:", question_embedding.size())

        # In: batch_size * passage_len
        # Out: batch_size * passage_len * word_embedding_dim
        passage_embedding = self.word_embedding(passage)
        # print("passage embedding size:", passage_embedding.size())
        # Out:
        # question_encoding: [batch_size,question_len,hidden_size*num_layers]
        # passage_encoding: [batch_size,passage_len,hidden_size*num_layers]
        question_encoding, passage_encoding = self.sentence_encoding(
                                            question_embedding, passage_embedding)
        #print("question encoding size:", question_encoding.size())
        # print("passage encoding size:", passage_encoding.size())

        # Out: a list of [batch_size, hidden_size*num_layers], len(list) = passage_len
        pair_encoding, _ = self.pair_encoder(question_encoding, passage_encoding)
        # Out: passage_len, batch_size, hidden_size*num_layers
        pair_encoding = torch.stack(pair_encoding, dim=0)
        pair_encoding = pair_encoding.transpose(0, 1)
        # print("pair_encoding size:", pair_encoding.size())
        # Out: [batch_size, passage_len, hidden_size*num_layers]
        self_matched_encoding, _ = self.self_matching_encoder(pair_encoding)
        self_matched_encoding = self_matched_encoding.transpose(0, 1)
        # print("self_matched_encoding size:", self_matched_encoding.size())
        # print("------------pointer network---------------")

        begin, end = self.pointer_output(question_encoding, self_matched_encoding)
        # print("begin size:", begin.size())
        # print("end size:", end.size())
        return begin, end
