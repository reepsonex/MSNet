import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class LocalConv2dReLU(nn.Module):
    def __init__(self, local_h_num, local_w_num, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, activation_type='ReLU'):
        super(LocalConv2dReLU, self).__init__()
        self.local_h_num = local_h_num
        self.local_w_num = local_w_num

        self.bns = nn.ModuleList([nn.BatchNorm2d(in_channels) for i in range(local_h_num * local_w_num)])

        if activation_type == 'ReLU':
            self.relus = nn.ModuleList([nn.ReLU(inplace=True) for i in range(local_h_num * local_w_num)])
        elif activation_type == 'PReLU':
            self.relus = nn.ModuleList([nn.PReLU() for i in range(local_h_num * local_w_num)])

        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias) for i in range(local_h_num * local_w_num)])

    def forward(self, x):
        h_splits = torch.split(x, int(x.size(2) / self.local_h_num), 2)

        h_out = []
        for i in range(len(h_splits)):
            start = True
            w_splits = torch.split(h_splits[i], int(h_splits[i].size(3) / self.local_w_num), 3)
            for j in range(len(w_splits)):
                bn_out = self.bns[i * len(w_splits) + j](w_splits[j].contiguous())
                bn_out = self.relus[i * len(w_splits) + j](bn_out)
                conv_out = self.convs[i * len(w_splits) + j](bn_out)
                if start:
                    h_out.append(conv_out)
                    start = False
                else:
                    h_out[i] = torch.cat((h_out[i], conv_out), 3)
            if i == 0:
                out = h_out[i]
            else:
                out = torch.cat((out, h_out[i]), 2)

        return out

class HierarchicalMultiScaleRegionLayer(nn.Module):
    def __init__(self, local_group, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, activation_type='ReLU'):
        super(HierarchicalMultiScaleRegionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)
        self.local_conv_branch1 = LocalConv2dReLU(local_group[0][0], local_group[0][1], out_channels, int(out_channels / 2),
                                                  kernel_size, stride,
                                                  padding, dilation, groups, bias, activation_type)
        self.local_conv_branch2 = LocalConv2dReLU(local_group[1][0], local_group[1][1], out_channels,
                                                  int(out_channels / 2), kernel_size, stride,
                                                  padding, dilation, groups, bias, activation_type)

        self.bn = nn.BatchNorm2d(out_channels)

        if activation_type == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif activation_type == 'PReLU':
            self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        local_branch1 = self.local_conv_branch1(x)
        local_branch2 = self.local_conv_branch2(x)
        local_out = torch.cat((local_branch1, local_branch2), 1)

        out = x + local_out
        out = self.bn(out)
        out = self.relu(out)

        return out


