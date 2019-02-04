from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from torch4hep.modules import Activation
from torch4hep.utils.convolution import get_conv_same_padding

from .conv2d_same_pad import Conv2dSamePad


_ALLOWED_CONV_KEYS = (
    'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding',
    'dilation', 'groups', 'bias'
)

_ALLOWED_BATCH_NORM_KEYS = (
    'num_features', 'eps', 'momentum', 'affine', 'track_running_stats')


def _normalize_padding(padding,
                       kernel_size=1,
                       stride=1,
                       dilation=1,
                       **kwargs):
    del kwargs

    def _normalize_arg(x):
        if isinstance(x, Iterable):
            if x[0] == x[1] == 1:
                x = 1
        return x

    kernel_size = _normalize_arg(kernel_size)
    stride = _normalize_arg(stride)
    dilation = _normalize_arg(dilation)

    if isinstance(padding, str):
        padding = padding.lower()
        if padding == 'same':
            if kernel_size == 1 and stride == 1 and dilation == 1:
                padding = 0
            else:
                pass
        elif padding == 'valid':
            padding = 0
        elif padding == 'causal':
            raise NotImplementedError
        else:
            raise ValueError
    return padding


class Conv2dUnit(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding='same',
                 order='bac',
                 activation='relu',
                 activation_inplce=True,
                 dropout_rate=0.5,
                 dropout_inplace=False,
                 activation_kwargs={},
                 **kwargs):
        super(Conv2dUnit, self).__init__()

        # Filtering kwargs
        self.conv_kwargs = dict()
        self.batch_norm_kwargs = dict()
        for key in kwargs:
            if key in _ALLOWED_CONV_KEYS:
                self.conv_kwargs[key] = kwargs[key]
            elif key in _ALLOWED_BATCH_NORM_KEYS:
                self.batch_norm_kwargs[key] = kwargs[key]
            else:
                raise KeyError(key)

        if (not self.conv_kwargs.has_key('bias')) and ('b' in order):
            self.conv_kwargs['bias'] = False

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = _normalize_padding(padding, kernel_size, **self.conv_kwargs)
        self.order = order

        self.activation = activation
        self.activation_kwargs = activation_kwargs


        if 'd' in order:
            self.dropout_rate = dropout_rate
            self.dropout_inplace = dropout_inplace

        self.num_ops = {key: 0 for key in 'abcd'}
        for each in order:
            name, module = self._get_name_and_module(each)
            self.add_module(name, module)

    def _normalize_order(self, order, delimiter='-'):
        '''
        TODO
        'Conv2d-BatchNorm2d-Activation-Dropout'
        '''
        raise NotImplementedError

    def _get_name_and_module(self, symbol):
        self.num_ops[symbol] += 1

        if symbol == 'a':
            name = '{}_{}'.format(self.activation, self.num_ops[symbol])
            module = Activation(self.activation, **self.activation_kwargs)
        elif symbol == 'b':
            if self.order.find('b') < self.order.find('c'):
                num_features = self.in_channels
            else:
                num_features = self.out_channels
            name = 'batch_norm_2d_{}'.format(self.num_ops[symbol])
            module = nn.BatchNorm2d(num_features=num_features,
                                    **self.batch_norm_kwargs)
        elif symbol == 'c':
            name = 'conv_2d_{}'.format(self.num_ops[symbol])
            if self.padding == 'same':
                module = Conv2dSamePad(self.in_channels,
                                       self.out_channels,
                                       self.kernel_size,
                                       **self.conv_kwargs)
            else:
                module = nn.Conv2d(self.in_channels,
                                   self.out_channels,
                                   self.kernel_size,
                                   **self.conv_kwargs)
        elif symbol == 'd':
            name = 'dropout_{}'.format(self.num_ops[symbol])
            module = nn.Dropout(p=self.dropout_rate,
                                inplace=self.dropout_inplace)
        else:
            raise ValueError

        return name, module



def _test():
    x = torch.randn(128, 10, 33, 33)

    conv_unit = Conv2dUnit(
        order='bacd',
        in_channels=10,
        out_channels=20,
        kernel_size=3,
        padding=1,
        activation='elu',
        eps=1e-5,
        momentum=0.2,
        activation_kwargs={'alpha': 1.01})

    print(conv_unit)
       
    h = conv_unit(x)
    print(h.shape)

def main():
    _test()

if __name__ == '__main__':
    main()
