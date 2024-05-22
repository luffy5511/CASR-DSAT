import random
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np 
from seed_everything import seed_everything
import blocks as blocks
from DAST import TMENet
from collections import OrderedDict

seed_everything(0)
class ResBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, input):
        out = self.conv1(input)
        out = F.relu(out)
        out = self.conv2(out)
        out = input + out
        return out
    
class Conv_ResBlock(nn.Module):
    def __init__(self, in_dim, conv_dim):
        super(Conv_ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.ResBlock = ResBlock(conv_dim)

    def forward(self, input):
        out = self.conv1(input)
        out = F.relu(out)
        out = self.ResBlock(out)
        return out


######################### Encoder ##############################
class EncoderNet(nn.Module):
    def __init__(self, in_dim=1, conv_dim = 32, out_dim=32, num_blocks=4):
        super(EncoderNet, self).__init__()
        self.inputConv = nn.Conv2d(in_channels=in_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.ResBlocks = nn.Sequential(*[ResBlock(conv_dim) for i in range(num_blocks)])

        self.outputConv = nn.Conv2d(in_channels=conv_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, input):
        out = self.inputConv(input)
        out = self.ResBlocks(out)
        out = F.relu(out)
        out = self.outputConv(out)
        return out

########################## Decoder ############################
class DecoderNet(nn.Module):
    def __init__(self, in_dim=64, out_dim=1, num_blocks=10):
        super(DecoderNet, self).__init__()
        self.inputConv = nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.ResBlocks = nn.Sequential(*[ResBlock(64) for i in range(num_blocks)])
        self.outputConv = nn.Conv2d(in_channels=64, out_channels=out_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, input):
        out = self.inputConv(input)
        out = self.ResBlocks(out)
        #out = self.deconv1(out)
        #out = F.relu(out)
        #out = self.deconv2(out)
        out = F.relu(out)
        out = self.outputConv(out)
        return out
    


    
    
###########################################################################
class ConvLeaky(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvLeaky, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        #nn.init.kaiming_normal_(self.conv1.weight)
        #nn.init.kaiming_normal_(self.conv2.weight)
        
        
    def forward(self, input):
        out = self.conv1(input)
        #print('conv1: {}'.format(out.size()))
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = F.leaky_relu(out, 0.2)
        return out
    
class FNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, mode):
        super(FNetBlock, self).__init__()
        self.convleaky = ConvLeaky(in_dim, out_dim)
        if mode == "maxpool":
            self.final = lambda x: F.max_pool2d(x, kernel_size=2)
        elif mode == "bilinear":
            self.final = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            raise Exception('mode must be maxpool or bilinear')

    def forward(self, input):
        out = self.convleaky(input)
        out = self.final(out)
        return out

class Refiner(torch.nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()

        self.netMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3,
                            stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
        )

    def forward(self, tenInput):
        return self.netMain(tenInput)


class FNet(nn.Module):
    def __init__(self, in_dim=2):
        super(FNet, self).__init__()
        self.convPool1 = FNetBlock(in_dim, 32, mode="maxpool")
        self.convPool2 = FNetBlock(32, 64, mode="maxpool")
        self.convPool3 = FNetBlock(64, 128, mode="maxpool")
        self.convBinl1 = FNetBlock(128, 256, mode="bilinear")
        self.convBinl2 = FNetBlock(256, 128, mode="bilinear")
        self.convBinl3 = FNetBlock(128, 64, mode="bilinear")
        self.seq = nn.Sequential(self.convPool1, self.convPool2, self.convPool3,
                                 self.convBinl1, self.convBinl2, self.convBinl3)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        #self.refine = Refiner()

        
        #nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
    def forward(self, input):
        out = self.seq(input)
        out = self.conv1(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        #refineout = self.refine(torch.cat((input[:, 0:1, ...], out), dim=1)) # torch.cat((input[:, 1:2, ...], out), dim=1)
        #out = out + refineout
        self.out = torch.tanh(out)*10.0 # 原来10.0
        #self.out.retain_grad()
        return self.out


class TransformerME(nn.Module):
    def __init__(self, in_dim=2, img_size=56):
        super(TransformerME, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=32, kernel_size=3, stride=1, padding=1,
                               padding_mode='reflect')
        self.decoder = TMENet(inp_channels=32,
                                  out_channels=2,
                                  num_blocks=[2, 2, 2],
                                  heads=[2, 2, 2],
                                  img_size=img_size,
                                  bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.decoder(out)
        self.out = torch.tanh(out) * 8.0
        # self.out.retain_grad()
        return self.out

class Outlier_pred(nn.Module):
    def __init__(self, in_dim=1, project_dim=64, out_dim=1, num_weight_predictor_res=1):
        super(Outlier_pred, self).__init__()
        weight_predictor = []
        weight_predictor.append(blocks.conv_block(in_dim, project_dim, 3,
                                                  stride=1, padding=1, batch_norm=False, activation='relu'))

        for _ in range(num_weight_predictor_res):
            weight_predictor.append(blocks.ResBlock(project_dim, project_dim, stride=1,
                                                    batch_norm=False, activation='relu'))

        weight_predictor.append(blocks.conv_block(project_dim, out_dim, 3, stride=1, padding=1,
                                                  batch_norm=False,
                                                  activation='none'))

        self.weight_predictor = nn.Sequential(*weight_predictor)

    def forward(self, x):
        x = self.weight_predictor(x)
        x = torch.tanh(x)
        return x

