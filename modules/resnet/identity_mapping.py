from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F


class ZeroPadShortcut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ZeroPadShortcut, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=2)

        channel_diff = out_channels - in_channels
        pad0 = channel_diff // 2
        pad1 = channel_diff - pad0
        self.pad = (0, 0, 0, 0, pad0, pad1)

    def forward(self, input):
        output = self.avg_pool(input)
        return F.pad(input=output, pad=self.pad)


class ProjectionShortcut(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ProjectionShortcut, self).__init__()

        self.conv1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride)

        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels)
