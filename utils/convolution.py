'''arXiv:1603.07285

https://github.com/vdumoulin/conv_arithmetic

https://tensorflow.blog/a-guide-to-convolution-arithmetic-for-deep-learning/ (Korean)
https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sympy
import math
from collections import Iterable

from torch.nn.modules.utils import _pair


def allowing_pair_args(function):
    def wrapped_function(**kwargs):
        iterable_exists = any(isinstance(each, Iterable) for each in kwargs.values())
        if iterable_exists:
            kwargs = {key: _pair(value) for key, value in kwargs.iteritems()}
            zipped_kwargs = [{key: value[i] for key, value in kwargs.iteritems()} for i in range(2)]
            output = tuple(function(**each) for each in zipped_kwargs)
        else:
            output = function(**kwargs)
        return output
    return wrapped_function


@allowing_pair_args
def get_conv_out_length(in_length,
                        kernel_size,
                        stride=1,
                        padding=0, 
                        dilation=1):
    out_length = ((in_length + 2*padding - dilation*(kernel_size - 1) - 1) / stride) + 1 
    out_length = math.floor(out_length)
    return int(out_length)


@allowing_pair_args
def get_conv_transpose_out_length(in_length,
                                  kernel_size,
                                  stride=1,
                                  padding=0,
                                  out_padding=0):
    return (in_length - 1)*stride - 2*padding + kernel_size + out_padding


@allowing_pair_args
def get_conv_padding(in_length,
                     out_length,
                     kernel_size=1,
                     stride=1,
                     dilation=1):
    padding = sympy.symbols("p")

    # TODO write the link to equation
    numerator = in_length + 2*padding - dilation*(kernel_size-1) - 1
    right_side= (numerator / stride) + 1
    equation = out_length - right_side

    # choose the smallest solution
    padding = sympy.solve(equation)[0]
    padding = sympy.ceiling(padding)
    padding = int(padding)
    return padding


def get_conv_same_padding(in_length,
                          kernel_size=1,
                          stride=1,
                          dilation=1):
    padding = get_conv_padding(in_length=in_length,
                               out_length=in_length,
                               kernel_size=kernel_size,
                               stride=stride,
                               dilation=dilation)
    return padding
