'''
Stolen from https://github.com/tensorflow/models/blob/master/official/transformer/model/attention_layer.py

TODO cache
# TODO reset_parameter
# TODO key, value length != query length
# TODO check dim

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch.nn import init


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 num_heads,
                 padding_value=0,
                 negative_infinity=-1e9):
        super(MultiHeadAttention, self).__init__()

        assert output_size % num_heads == 0

        self.input_size = input_size,
        self.output_size = output_size
        self.num_heads = num_heads
        self.padding_value = padding_value
        self.negative_infinity = negative_infinity

        self.depth = int(output_size / num_heads)
        self.scale_factor = self.depth ** -0.5 

        def get_linear():
            return nn.Linear(
                in_features=input_size,
                out_features=output_size,
                bias=False)

        self.linear_key = get_linear() 
        self.linear_value = get_linear() 
        self.linear_query = get_linear() 

        self.linear_output = nn.Linear(
            in_features=output_size,
            out_features=output_size,
            bias=False)

        self.reset_parameters()
 
    def reset_parameters(self):
        init.xavier_uniform_(self.linear_key.weight)
        init.xavier_uniform_(self.linear_value.weight)
        init.xavier_uniform_(self.linear_query.weight)
        init.xavier_uniform_(self.linear_output.weight)

    def forward(self, key, value, query):
        key = self.linear_key(key)
        value = self.linear_value(value)
        query = self.linear_query(query)

        key = self.split_into_heads(key)
        value = self.split_into_heads(value)
        query = self.split_into_heads(query)

        key = key.permute([0, 1, 3, 2])
        query *= self.scale_factor
        # query = query * self.scale_factor

        logits = torch.matmul(query, key)
        logits = logits + self.get_bias(value)

        # TODO Check dim
        attention = torch.softmax(logits, dim=-1)

        output = torch.matmul(attention, value)
        output = self.combine_heads(output)
        return self.linear_output(output)

    def split_into_heads(self, tensor):
        batch_size, seq_len, _, = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, self.num_heads, self.depth)
        return tensor.permute([0, 2, 1, 3])

    def get_bias(self, tensor):
        is_padding_value = torch.eq(tensor, self.padding_value)
        is_padding = is_padding_value.all(dim=-1)
        is_padding = is_padding.unsqueeze(-1)
        is_padding = is_padding.float()
        bias = is_padding * self.negative_infinity
        return bias

    def combine_heads(self, tensor):
        batch_size, _, seq_len, _ = tensor.shape
        tensor = tensor.permute([0, 2, 1, 3])
        return tensor.reshape(batch_size, seq_len, self.output_size)


class MultiHeadSelfAttention(MultiHeadAttention):
    def forward(self, input):
        return super(MultiHeadSelfAttention, self).forward(input, input, input)
