import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence


class StackedCell(nn.Module):
    """
    From https://github.com/eladhoffer/seq2seq.pytorch
    MIT License  Copyright (c) 2017 Elad Hoffer
    """
    def __init__(self, input_size, hidden_size, num_layers=1,
                 dropout=0, bias=True, rnn_cell=nn.GRUCell, residual=False):
        super(StackedCell, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.residual = residual
        self.layers = nn.ModuleList()
        # print("stacked cell num_layers:", num_layers)
        for _ in range(num_layers):
            # print("rnn_cell, input_size:", input_size)
            # print("rnn_cell, hidden_size:", hidden_size)
            self.layers.append(rnn_cell(input_size, hidden_size, bias=bias))
            input_size = hidden_size

    def forward(self, inputs, hidden):
        def select_layer(h_state, i):  # To work on both LSTM / GRU, RNN
            if isinstance(h_state, tuple):
                return tuple([select_layer(s, i) for s in h_state])
            else:
                return h_state[i]

        next_hidden = []
        
        for i, layer in enumerate(self.layers):
            # print(layer)
            hidden_i = select_layer(hidden, i)
            # print("cell inputs size:", inputs.size())
            # print("cell hidden_i size:", hidden_i.size())
            next_hidden_i = layer(inputs, hidden_i)
            output = next_hidden_i[0] if isinstance(next_hidden_i, tuple) \
                else next_hidden_i
            if i + 1 != self.num_layers:
                output = self.dropout(output)
            if i > 0 and self.residual:
                inputs = output + inputs
            else:
                inputs = output
            next_hidden.append(next_hidden_i)
        if isinstance(hidden, tuple):
            next_hidden = tuple([torch.stack(h) for h in zip(*next_hidden)])
        else:
            next_hidden = torch.stack(next_hidden)
        return inputs, next_hidden


class AttentionEncoderCell(StackedCell):
    def __init__(self, question_embed_size, passage_embed_size, hidden_size,
                 attention_layer_factory, attn_args, attn_kwags, attn_mode="pair_encoding", num_layers=1,
                 dropout=0, bias=True, rnn_cell=nn.GRUCell, residual=False,
                 gated=True):
        input_size = question_embed_size + passage_embed_size
        super().__init__(input_size, hidden_size, num_layers,
                         dropout, bias, rnn_cell, residual)
        self.attention = attention_layer_factory(*attn_args, **attn_kwags)
        self.gated = gated
        self.attn_mode = attn_mode
        if gated:
            self.gate = nn.Sequential(
                nn.Linear(input_size, input_size, bias=False),
                nn.Sigmoid()
            )

    def forward(self, input_with_context, hidden):
        inputs, context = input_with_context

        if isinstance(hidden, tuple):
            hidden_for_attention = hidden[0]
        else:
            hidden_for_attention = hidden
        hidden_for_attention = hidden_for_attention[0:1]
                
        key = context
        if self.attn_mode == "pair_encoding":
            hidden_for_attention = hidden_for_attention.transpose(0, 1) 
            # print("u_t^P:", inputs.size())
            # print("v_(t-1)^P:", hidden_for_attention.size())
            queries = [inputs, hidden_for_attention]
        elif self.attn_mode == "self_matching":
            queries = [inputs]
        else:
            raise ValueError("invalid attention_mode %s" % self.attn_mode)
		
        context = self.attention(key, queries)
        inputs = torch.cat([context, inputs], dim=context.dim()-1)
        if self.gated:
            inputs = inputs * self.gate(inputs)
        return super().forward(inputs.squeeze(0), hidden)


class AttentionEncoder(nn.Module):
    def __init__(self, cell_factory, *args, bidirectional=False, mode="GRU", **kwargs):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = args[2]
        self.forward_cell = cell_factory(*args, **kwargs)
        self.num_layers = self.forward_cell.num_layers
        self.mode=mode
        if bidirectional:
            self.reversed_cell = cell_factory(*args, **kwargs)

    def _forward(self, inputs, hidden):
        query, context = inputs
        output = []

        query_len = query.size(1)
        # batch_size*query_len*n
        # -> query_len*batch_size*n
        query = query.transpose(0, 1)
        # print("hidden size:", hidden.size())
        for i in range(query_len):
            step_input = query[i].unsqueeze(1) # batch_size*1*n
            hidden = self.forward_cell((step_input, context), hidden)[1]
            output.append(hidden[0])

        return output, hidden[0]

    def _reversed_forward(self, inputs, hidden):
        query, context = inputs

        output = []

        query_len = query.size(1)
        # batch_size*query_len*n
        # -> query_len*batch_size*n
        query = query.transpose(0, 1)
        
        initial_hidden = hidden
        for i in range(query_len):
            step_input = query[query_len-i-1].unsqueeze(1)  # batch_size*1*n
            hidden = self.reversed_cell((step_input, context), hidden)[1]
            output.append(hidden[0])

        output.reverse()
        return output, hidden[0]

    def forward(self, inputs, hidden=None):
        query, context = inputs
        if hidden is None:
            batch_size = query.size(0)
            # print(batch_size)
            hidden = torch.autograd.Variable(torch.zeros(self.num_layers,
                                                       batch_size,
                                                       self.hidden_size))

            if torch.cuda.is_available():
                hidden = hidden.cuda()

        output_forward, hidden_forward = self._forward(inputs, hidden)
        if not self.bidirectional:
            return output_forward, hidden_forward
        output_reversed, hidden_reversed = self._reversed_forward(inputs, hidden)

        # concat forward and reversed forward
        hidden = torch.cat([hidden_forward, hidden_reversed], dim=hidden_forward.dim() - 1)
        output_forward = torch.stack(output_forward, dim=0)
        # print("output forward size: ", output_forward.size())
        output_reversed = torch.stack(output_reversed, dim=0)
        # print("output reversed size: ", output_reversed.size())
        output = torch.cat([output_forward, output_reversed],
                                dim=output_reversed.data.dim() - 1)
        # print("output size:",output.size())
        return output, hidden
