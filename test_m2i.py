
import datetime
import random
import numpy as np
import os
import argparse
import tempfile
from time import time

from numpy import mean
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from MEnet import FNet
from DAST import EncoderNet, DecoderNet, DecoderNet2
from shiftandadd import shiftAndAdd, featureAdd2, featureWeight
from HRPAF import PCAFNet, AGFNet
from warpingOperator import WarpedLoss, TVL1, base_detail_decomp, BlurLayer
import os
from torch.autograd import Variable
from synthetic_dataset.zurich_raw2rgb_dataset import ZurichRAW2RGB
from synthetic_dataset.synthetic_burst_train_set import SyntheticBurst

from torchvision.transforms import GaussianBlur
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import cv2
from seed_everything import seed_everything
from loss import Pyramid, PSNRM
from collections import OrderedDict
from real_dataset.burstsr_dataset import BurstSRDataset, BurstSRDataset2


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.makedirs(path)
    except OSError:
        pass


def flowEstimation(samplesLR, ME, c, warping, gaussian_filter, losstype):
    """
    Compute the optical flows from the other apertures to the reference aperture:
    samplesLR: Tensor b, num_im, h, w
    ME: Motion Estimator
    """

    b, num_im, h, w = samplesLR.shape

    samplesLRblur = gaussian_filter(samplesLR)  # Denoise and remove aliasing

    samplesLR_0 = samplesLRblur[:, :1, ...]  # b, 1, h, w

    samplesLR_0 = samplesLR_0.repeat(1, num_im, 1, 1)  # b, num_im, h, w
    samplesLR_0 = samplesLR_0.reshape(-1, h, w)
    samplesLRblur = samplesLRblur.reshape(-1, h, w)  # b*num_im, h, w
    samplesLRblur = samplesLRblur.unsqueeze(1)
    samplesLR_0 = samplesLR_0.unsqueeze(1)
    concat = torch.cat((samplesLRblur, samplesLR_0), dim=1)  # b*(num_im), 2, h, w
    flow = ME(concat.cuda())  # b*(num_im), 2, h, w
    flow[::num_im] = 0

    warploss, warped = warping(samplesLRblur, samplesLR_0, flow, c, losstype=losstype)
    # warploss: alignment loss

    return flow.reshape(b, num_im, 2, h, w), warploss, warped


class DeepSR(nn.Module):
    def __init__(self, num_im=9, Encoder_num_features=64, PCAF_num_features=64, sr_ratio=3, img_size=(192, 192)):
        super(DeepSR, self).__init__()
        self.encoder = EncoderNet(inp_channels=1,
                                  out_channels=Encoder_num_features,
                                  num_blocks=[1, 2, 4],
                                  heads=[1, 2, 2],
                                  ffn_expansion_factor=1.5,  # Expansion coefficient in gate module
                                  bias=False,
                                  LayerNorm_type='WithBias')
        # EncoderNet: feature extraction backbone based on channel self-attention transformer block
        self.PCAF = PCAFNet(num_im, PCAF_num_features)
        # PCAFNet: pseudo camera array feature
        self.AGF = AGFNet(PCAF_num_features, 4)
        # AGFNet: high-resolution adaptive group fusion
        self.decoder = DecoderNet(inp_channels=PCAF_num_features,
                                  out_channels=1,
                                  num_blocks=[2, 2, 4],
                                  heads=[2, 2, 4],
                                  img_size=img_size,
                                  bias=False)
        # DecoderNet: feature reconstruction based on spatial self-attention transformer block
        self.MEAlign = FNet().float()
        # FNet: convolutional network for motion estimation
        self.sr_ratio = sr_ratio
        self.Encoder_num_features = Encoder_num_features

    def forward(self, samplesLR, flag_train, c, losstype):
        flow, warploss, warped = flowEstimation(samplesLR, self.MEAlign, c,
                                                WarpedLoss(interpolation='bicubicTorch'),
                                                gaussian_filter=GaussianBlur(11, sigma=1), losstype=losstype)
        b, num_im, h, w = samplesLR.shape
        samplesLR = samplesLR.view(-1, 1, h, w)  # b*(num_im),1, h, w
        if flag_train:
            samplesLR[::num_im, ...] = torch.tensor(0.0).cuda()
        features = self.encoder(samplesLR).view(-1, h, w)  # b * (num_im), num_features, h, w
        flowf = flow.contiguous().view(-1, 1, 2, h, w).repeat(1, self.Encoder_num_features, 1, 1, 1).view(-1, 2, h, w)  # b * num_im* num_features, 2, h, w
        dadd = featureAdd2(features, flowf, sr_ratio=self.sr_ratio)  # b * num_im * num_features, sr_ration*h, sr_ratio*w
        # featureAdd: Sub-pixel motion compensation without parameters
        dadd = dadd.view(b, num_im, self.Encoder_num_features, self.sr_ratio * h, self.sr_ratio * w)
        SR = self.AGF(self.PCAF(dadd)).squeeze(
            1)  # b, 1, PCAF_num_features, sr_ratio * h, sr_ratio * w
        SR = self.decoder(SR)  # b, 1, sr_ration*h, sr_ratio*w

        return SR, flow, warploss, warped


def BackWarping(x, flo, ds_factor=3):
    """
    backward warp and downsample an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    if torch.sum(flo * flo) == 0:
        return x[..., ::ds_factor, ::ds_factor]
    else:
        B, _, H, W = flo.size()

        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.cuda()
        # print(grid.shape)
        vgrid = ds_factor * (Variable(grid) + flo)
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(ds_factor * W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(ds_factor * H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        output = nn.functional.grid_sample(x, vgrid, align_corners=True, mode='bicubic', padding_mode='reflection')

        return output


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt

"""Test synthetic datasets using pretrained CASR-DAST model"""
def test(args):
    seed_everything(0)
    # Load the parameters
    val_bs, num_im, sr_ratio, val_patchsize = \
        args.val_bs, args.num_im, args.sr_ratio, args.val_patchsize
    Ed_num_features, PCAF_num_features, num_blocks = args.Ed_num_features, args.PCAF_num_features, args.num_blocks

    sr_ratio = args.sr_ratio
    folder_name = 'self-supervised_multi-image_deepSR_time_{}'.format(
        f"{datetime.datetime.now():%m-%d-%H-%M-%S}")

    ################## load Models
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for testing.")
    deepSuperresolve = DeepSR(num_im=num_im,
                              Encoder_num_features=Ed_num_features, PCAF_num_features=PCAF_num_features,
                              sr_ratio=sr_ratio, img_size=(val_patchsize*sr_ratio, val_patchsize*sr_ratio)).cuda()

    checkpoint_path ="/home/WXS/CYT/TrainHistory/self-supervised_multi-image_deepSR_time_11-19-22-40-51/checkpoint_12.pth.tar"
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict DeepSR']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    deepSuperresolve.load_state_dict(new_state_dict)
    Dataset_path = "/home/WXS/CYT/DBSR/test_synthetic_burst/DIV2K_valid_HR/"
    val_data_set = BurstSRDataset2(root=Dataset_path, split='test', burst_size=9, crop_sz=(336, 336), center_crop=True,
                                   random_flip=False)
    val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=val_bs,
                                             num_workers=1, shuffle=False)
    result_dir_SR ="/home/WXS/CYT/CASR/消融实验/ours/test_synthetic_SR/DIV2K_valid_HR/"
    ##################
    starttime = time()
    ##################
    with torch.no_grad():
        for k, data in enumerate(val_loader):
            samplesLR= data
            b, num_im, h, w = samplesLR.shape
            samplesLR = samplesLR.float().cuda()
            c = 8  # partially overlapping border
            SR, flow, warploss, warped = deepSuperresolve(samplesLR.clone(), False, c, 'Detail') # Reference LR image remains unchanged during testing
            SR = SR.detach().cpu().numpy().squeeze()
            cv2.imwrite(os.path.join(result_dir_SR, "SR_{:03d}.png".format(k)), SR)
            '''warped = warped.view(b, num_im, h, w)
            for n in range(num_im):
                LR = samplesLR[0, n, ...]
                LR = LR.detach().cpu().numpy().squeeze()
                cv2.imwrite(os.path.join(result_dir_SR, "LR_{:03d}_{:02d}.png".format(k, n)), LR)
                warp = warped[0, n, ...]
                warp = warp.detach().cpu().numpy().squeeze()
                cv2.imwrite(os.path.join(result_dir_SR, "Warped_{:03d}_{:02d}.png".format(k, n)), warp)'''

    print('Execution time = {:.0f}s'.format(time() - starttime))
    return

def main(args):

    torch.cuda.empty_cache()
    test(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-on", "--option_name", help="Option id", default='J_selfSR')
    parser.add_argument("-bsv", "--val_bs", help="Batch size of val loader", type=int, default=1)  # 5
    parser.add_argument("-ednf", "--Ed_num_features", help="Num of features for encoder", type=int, default=64)
    parser.add_argument("-PCAFnf", "--PCAF_num_features", help="Num of features for PCAF", type=int, default=64)
    parser.add_argument("-nb", "--num_blocks", help="Number of residual blocks in encoder", type=int, default=4)
    parser.add_argument("-srr", "--sr_ratio", help="Super-resolution factor", type=int, default=3)
    parser.add_argument('-num', '--num_im', nargs='+', help="Number of image for camera array", default=9)
    parser.add_argument('-vps', '--val_patchsize', help="the size of crop for val", default=168)

    args = parser.parse_args()

    main(args)