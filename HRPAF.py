import torch

import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import numpy as np
import random
import os
from seed_everything import seed_everything
from DAST import TransformerBlock

seed_everything(0)


##############################################################################################
######################### Residual Global Context Attention Block ##########################################
##############################################################################################

class RGCAB(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RGCAB, self).__init__()
        self.module = [RGCA(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)

class RGCA(nn.Module):
    def __init__(self, n_feat, reduction=8, bias=False, act=nn.LeakyReLU(negative_slope=0.2,inplace=True), groups =1):

        super(RGCA, self).__init__()

        self.n_feat = n_feat
        self.groups = groups
        self.reduction = reduction

        modules_body = [nn.Conv2d(n_feat, n_feat, 3,1,1 , bias=bias, groups=groups), act, nn.Conv2d(n_feat, n_feat, 3,1,1 , bias=bias, groups=groups)]
        self.body   = nn.Sequential(*modules_body)

        self.gcnet = nn.Sequential(GCA(n_feat, n_feat))
        self.conv1x1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.gcnet(res)
        res = self.conv1x1(res)
        res = res + x
        return res


######################### Global Context Attention ##########################################

class GCA(nn.Module):
    def __init__(self, inplanes, planes, act=nn.LeakyReLU(negative_slope=0.2,inplace=True), bias=False):
        super(GCA, self).__init__()

        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias),
            act,
            nn.Conv2d(planes, inplanes, kernel_size=1, bias=bias)
        )

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x

##############################################################################################
######################### Multi-scale Feature Extractor ##########################################
##############################################################################################

class UpSample(nn.Module):

    def __init__(self, in_channels, chan_factor, bias=False):
        super(UpSample, self).__init__()

        self.up = nn.Sequential(nn.Conv2d(in_channels, int(in_channels/chan_factor), 1, stride=1, padding=0, bias=bias),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    def forward(self, x):
        x = self.up(x)
        return x

class DownSample(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(DownSample, self).__init__()

        self.down = nn.Sequential(nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
                                nn.Conv2d(in_channels, int(in_channels*chan_factor), 1, stride=1, bias=bias))

    def forward(self, x):
        x = self.down(x)
        return x


class TSF(nn.Module):
    def __init__(self, in_channels=32, num_block=3, num_heads=2, ffn_expansion_factor=1.5, LayerNorm_type='WithBias', bias=False):
        super(TSF, self).__init__()
        self.TransformerGroup = nn.Sequential(*[
            TransformerBlock(dim=in_channels, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_block)])
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, x):
        x = x + self.conv(self.TransformerGroup(x))
        return x

class PCAFNet(nn.Module):
    def __init__(self, num_im, num_features=64, reduction=8, bias=False):
        super(PCAFNet, self).__init__()
        ####### Pseudo Burst Feature Fusion
        self.conv2 = nn.Conv2d(num_im, num_features, kernel_size=3, padding=1, bias=bias)
        self.num_features = num_features

    def forward(self, cc_feat):
        b, num_im, f, h, w = cc_feat.shape
        cc_feat = cc_feat.permute(0,2,1,3,4).contiguous() # 0,2,1,3,4
        cc_feat = cc_feat.view(b*f, num_im, h, w)
        cc_feat = self.conv2(cc_feat)       # (b*num_features, num_features, 2h, 2w)
        cc_feat = cc_feat.view(b, f, self.num_features, h, w)
        return cc_feat






######################### Adaptive Group Module ##########################################
class AGUNet(nn.Module):
    def __init__(self, in_channels, height, reduction=8, bias=False):
        super(AGUNet, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=False))

        self.convs = nn.ModuleList([])
        for i in range(self.height):
            self.convs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)
        self.conv_1 = nn.Conv2d(in_channels*4, in_channels, kernel_size=1, bias=bias)

    def forward(self, inp_feats):
        b, n, c, n_feats, h, w = inp_feats.size()
        inp_feats = inp_feats.view(b*n, c, n_feats, h, w)

        feats_U = torch.sum(inp_feats, dim=1)
        feats_Z = self.conv_du(feats_U)

        dense_attention = [conv(feats_Z) for conv in self.convs]
        dense_attention = torch.cat(dense_attention, dim=1)

        dense_attention = dense_attention.view(b*n, self.height, n_feats, h, w)

        dense_attention = self.softmax(dense_attention)

        feats_V = inp_feats * dense_attention
        feats_V = feats_V.view(b*n, -1, h, w)
        feats_V = self.conv_1(feats_V)  # 无需上采样
        feats_V = feats_V.view(b, n, n_feats, h, w)

        return feats_V

######################### Adaptive and Progressive Group Module ##########################################
class AGFNet(nn.Module):
    def __init__(self, in_channels, height, reduction=8, bias=False):
        super(AGFNet, self).__init__()
        self.height = height

        self.SKFF1 = AGUNet(in_channels, height, reduction, bias) #每一层的自适应分组上采样参数是不同的，同一层不同组参数是共享的
        self.SKFF2 = AGUNet(in_channels, height, reduction, bias)
        self.SKFF3 = AGUNet(in_channels, height, reduction, bias)

    def forward(self, in_feat):
        b, f1, f2, h, w = in_feat.size()  # num_features 默认是32通道
        in_feat = in_feat.view(b, f1 // self.height, self.height, f2, h, w)  # (num_features//4, 4, num_features, H, W)
        in_feat = self.SKFF1(in_feat)  # (num_features//4, num_features, H, W)

        b, f1, f2, h, w = in_feat.size()
        in_feat = in_feat.view(b, f1 // self.height, self.height, f2, h, w)  # (num_features//16, 4, num_features, H, W)
        in_feat = self.SKFF2(in_feat)  # (num_features//16, num_features, 2H, 2W)
        #b, f1, f2, h, w = in_feat.size()
        b, f1, f2, h, w = in_feat.size()
        in_feat = in_feat.view(b, f1 // self.height, self.height, f2, h, w)  # (1, 4, num_features, H, W)
        in_feat = self.SKFF3(in_feat) # (1, num_features, H, W)
        return in_feat #.view(b, 1, f1*f2, h, w)

