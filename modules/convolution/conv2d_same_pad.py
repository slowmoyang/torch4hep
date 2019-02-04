from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import torch
from torch import nn
import torch.nn.functional as F

from torch4hep.modules import Activation
from torch4hep.utils.convolution import get_conv_same_padding


class Conv2dSamePad(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True):
        super(Conv2dSamePad, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=0, dilation=dilation, groups=groups, bias=bias)

        self.padding = None

    def forward(self, input):
        if self.padding is None:
            h_pad, w_pad = get_conv_same_padding(
                in_length=tuple(input.shape[2:]),
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation)
            self.padding = (h_pad, w_pad)
            
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding is None:
            s += ", padding='same'"
        else:
            s += ", padding={}('same')".format(self.padding)

        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)




def _test():
    raise NotImplementedError

def main():
    _test()

if __name__ == '__main__':
    main()
