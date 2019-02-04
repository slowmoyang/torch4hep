# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


class PositionWiseFFN(nn.Module):
    def __init__(self,
                 input_size,
                 filter_size,
                 output_size,
                 dropout_rate=0.5,
                 padding_value=0,
                 epsilon=1e-6):
        super(PositionWiseFFN, self).__init__()

        self.input_size = input_size
        self.filter_size = filter_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.padding_value = padding_value
        self.epsilon = epsilon

        self.linear_filter = nn.Linear(input_size, filter_size, bias=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_output = nn.Linear(filter_size, output_size, bias=True)

        # self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(
            tensor=self.linear_filter.weight,
            gain=nn.init.calculate_gain('relu'))

        init.xavier_uniform_(
            tensor=self.linear_output.weight,
            gain=nn.init.calculate_gain('relu'))

        init.constant_(self.linear_filter.bias, 0.0)
        init.constant_(self.linear_output.bias, 0.0)

    def forward(self, input):
        batch_size, seq_len, input_dim = input.shape

        is_padding_value = torch.eq(input, self.padding_value)
        is_padding = is_padding_value.all(dim=-1).view(-1)
        non_pad_indices = torch.le(is_padding, self.epsilon).byte()
        mask = non_pad_indices.unsqueeze(-1)


        # In: (batch_size, seq_len, input_dim)
        # Out: (1, batch_size * seq_len, input_dim)
        input = input.view(-1, input_dim).unsqueeze(0)
        masked_input = input.masked_select(mask).view(-1, input_dim).unsqueeze(0)

        masked_output = self.linear_filter(masked_input)
        masked_output = F.relu(masked_output)

        masked_output = self.dropout(masked_output)

        masked_output = self.linear_output(masked_output)
        masked_output = F.relu(masked_output)

        masked_output = masked_output.squeeze(0)

        # Scattering
        mask = mask.expand(mask.shape[0], self.output_size)
        output = torch.zeros(batch_size * seq_len, self.output_size).to(input.device)
        output = output.masked_scatter(
            mask=mask,
            tensor=masked_output)
        output = output.view(batch_size, seq_len, self.output_size)

        return output
