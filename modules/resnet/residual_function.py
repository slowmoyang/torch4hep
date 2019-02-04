from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

from torch4hep.modules import Conv2dUnit


class Residual(nn.Sequential):
    '''full pre-activation residualll function
    arXiv:1603.05027
    '''
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample_stride = 2 if in_channels < out_channels else 1

        self.conv_unit_0 = Conv2dUnit(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=self.downsample_stride,
            padding=0,
            activation='relu',
            order='bac')

        self.conv_unit_1 = Conv2dUnit(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding='same',
            activation='relu',
            order='bac')


class Bottleneck(nn.Sequential):
    '''Residual function for full pre-activation'''

    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bottleneck_channels = out_channels // 4
        self.downsample_stride = 2 if in_channels < out_channels else 1

        self.conv_unit_0 = Conv2dUnit(
            in_channels=in_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding='same',
            order='bac')

        self.conv_unit_1 = Conv2dUnit(
            in_channels=self.bottleneck_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=3,
            stride=self.downsample_stride,
            padding=0,
            order='bac')

        self.conv_unit_2 = Conv2dUnit(
            in_channels=self.bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding='same',
            order='bac')
