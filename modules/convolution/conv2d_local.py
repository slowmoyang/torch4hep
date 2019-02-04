# coding=utf-8
# I refer to https://gist.github.com/guillefix/23bff068bdc457649b81027942873ce5.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair


from torch4hep.utils.convolution import get_conv_out_length


class Conv2dLocal(Module):
    def __init__(self,
                 in_size,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 dilation=1):
        """

        """
        super(Conv2dLocal, self).__init__()

        in_rank = len(in_size)
        if in_rank == 3:
            start_index = 0
        elif in_rank == 4:
            start_index = 1
        else:
            raise ValueError()

        self.in_channels = in_size[start_index]
        self.in_height = in_size[start_index + 1]
        self.in_width = in_size[start_index + 2]
        self.out_channels = out_channels

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
 
        self.out_height = get_conv_out_length(
            in_length=self.in_height,
            kernel_size=self.kernel_size[0],
            stride=self.stride[0],
            padding=self.padding[0],
            dilation=self.dilation[0])

        self.out_width = get_conv_out_length(
            in_length=self.in_width,
            kernel_size=self.kernel_size[1],
            stride=self.stride[1],
            padding=self.padding[1],
            dilation=self.dilation[1])

        self.weight = Parameter(
            torch.Tensor(
                self.out_height,
                self.out_width,
                self.out_channels,
                self.in_channels,
                *self.kernel_size))

        if bias:
            self.bias = Parameter(
                torch.Tensor(
                    self.out_channels,
                    self.out_height,
                    self.out_width))
        else:
            self.register_parameter('bias', None)
 
        self.reset_parameters()
 
    def reset_parameters(self):
        # # of parameters = # of channels * # of params per each channel
        num_params = self.in_channels * np.prod(self.kernel_size)
        stdv = 1.0 / math.sqrt(num_params)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return conv2d_local(
            input, self.weight, self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation)


def conv2d_local(input,
                 weight,
                 bias=None,
                 padding=0,
                 stride=1,
                 dilation=1):

    if input.dim() != 4:
        raise NotImplementedError(
            "Input Error: Only 4D input Tensors supported (got {}D)".format(input.dim()))

    if weight.dim() != 6:
        # out_height x out_width x out_channels x in_channels x kernel_height x kernel_width
        raise NotImplementedError(
            "Input Error: Only 6D weight Tensors supported (got {}D)".format(weight.dim()))

    batch_size = input.size()[0]

    weight_size = weight.size()

    out_height = weight_size[0]
    out_width = weight_size[1]
    out_channels = weight_size[2]
    in_channels = weight_size[3]
    kernel_height = weight_size[4]
    kernel_width = weight_size[5]

    kernel_size = (kernel_height, kernel_width)

    print("\nInput: {}".format(input.size()))

    # cols: [BatchSize,
    #        C_in * H_k * W_k,
    #        H_out * W_out]
    cols = F.unfold(input=input,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    stride=stride)

    print("cols = F.unflod(input): {}".format(cols.size()))
    print("C_in * H_k * W_k = {} * {} * {} = {}".format(
        in_channels,
        kernel_height,
        kernel_width,
        in_channels * kernel_height * kernel_width))

    print("H_out * W_out = {} * {} = {}".format(
        out_height,
        out_width,
        out_height * out_width))

    # [N, Pi(k), L, 1]
    cols = cols.unsqueeze(-1)
    print("cols.unsqueeze(-1): {}".format(cols.size()))

    # [N, L, 1, Pi(k)]
    cols = cols.permute(0, 2, 3, 1)
    print("cols.permute(0, 2, 3, 1): {}".format(cols.size()))

    # (H_out * H_out,
    #  C_out,        
    #  C_in * H_kernel * W_kernel)
    print("\nweight: {}".format(weight.size()))
    weight = weight.view(out_height * out_width,
                         out_channels,
                         in_channels * kernel_height * kernel_width)
    print("weight.view: {}".format(weight.size()))
    # (H_out * W_out,
    #  C_in * H_kernel * W_kernel,
    #  C_out)
    weight = weight.permute(0, 2, 1)
    print("weight.permute: {}".format(weight.size()))

    # cols: [N, L, 1, Pi(k)]
    # weight: [H_out * W_out, C_in * H_kernel * W_kernel, C_out]
    print("\nFor matmul")
    print("cols: {}".format(cols.shape))
    print("weight: {}".format(weight.shape))
    out = torch.matmul(cols, weight)

    out = out.view(batch_size, out_height, out_width, out_channels)
    out = out.permute(0, 3, 1, 2)

    if bias is not None:
        out = out + bias.expand_as(out)
    return out
