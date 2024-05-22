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
from DAST import EncoderNet, DecoderNet
from shiftandadd import shiftAndAdd, featureAdd, featureWeight
from HRPAF import PCAFNet, AGFNet
from warpingOperator import WarpedLoss, TVL1, base_detail_decomp, BlurLayer
import os
from torch.autograd import Variable
from synthetic_dataset.zurich_raw2rgb_dataset import ZurichRAW2RGB
from synthetic_dataset.synthetic_burst_train_set import SyntheticBurst
from real_dataset.burstsr_dataset import BurstSRDataset

from torchvision.transforms import GaussianBlur
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import cv2
from seed_everything import seed_everything
from loss import Pyramid, PSNRM
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.makedirs(path)
    except OSError:
        pass



def flowEstimation(samplesLR, local_rank, ME, c, warping, gaussian_filter, losstype):
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
    flow = ME(concat.cuda(local_rank))  # b*(num_im), 2, h, w
    flow[::num_im] = 0

    warploss, warped = warping(samplesLRblur, samplesLR_0, flow, c, losstype=losstype)
    # warploss: alignment loss

    return flow.reshape(b, num_im, 2, h, w), warploss, warped


class DeepSR(nn.Module):
    def __init__(self, local_rank, num_im=9, Encoder_num_features=64, PCAF_num_features=64, sr_ratio=3, img_size=(192, 192)):
        super(DeepSR, self).__init__()
        self.encoder = EncoderNet(inp_channels=1,
                                  out_channels=Encoder_num_features,
                                  num_blocks=[1, 2, 4],
                                  heads=[1, 2, 2],
                                  ffn_expansion_factor=1.5, # Expansion coefficient in gate module
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
        self.local_rank = local_rank
        self.Encoder_num_features = Encoder_num_features

    def forward(self, samplesLR, flag_train, c, losstype):
        flow, warploss, warped = flowEstimation(samplesLR, self.local_rank, self.MEAlign, c,
                                                WarpedLoss(interpolation='bicubicTorch'),
                                                gaussian_filter=GaussianBlur(11, sigma=1), losstype=losstype)
        b, num_im, h, w = samplesLR.shape
        samplesLR = samplesLR.view(-1, 1, h, w)  # b*(num_im),1, h, w

        if flag_train:
            samplesLR[::num_im, ...] = torch.tensor(0.0).cuda(self.local_rank)
            random_shifts = torch.randint(low=0, high=self.sr_ratio,
                                          size=(b, 1, 2, 1, 1)) / self.sr_ratio  # Grid shifting
            flow = flow - random_shifts.cuda(self.local_rank)

        features = self.encoder(samplesLR).view(-1, h, w)  # b * (num_im), num_features, h, w
        flowf = flow.contiguous().view(-1, 1, 2, h, w).repeat(1, self.Encoder_num_features, 1, 1, 1).view(-1, 2, h,w)  # b * num_im* num_features, 2, h, w
        dadd = featureAdd(features, flowf, sr_ratio=self.sr_ratio,
                          local_rank=self.local_rank)  # b * num_im * num_features, sr_ratio * h, sr_ratio * w
        # featureAdd: Sub-pixel motion compensation without parameters
        dadd = dadd.view(b, num_im, self.Encoder_num_features, self.sr_ratio * h, self.sr_ratio * w)
        SR = self.AGF(self.PCAF(dadd)).squeeze(
            1)  # b, 1, PCAF_num_features, sr_ratio * h, sr_ratio * w
        SR = self.decoder(SR)  # b, 1, sr_ration*h, sr_ratio*w

        return SR, flow, warploss, warped


def BackWarping(x, local_rank, flo, ds_factor=3):
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
        grid = grid.cuda(local_rank)
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

""" Fine-tuning the CSAR-DAST network using real camera array training dataset and pre-trained model"""
def train(local_rank, world_size, args):
    seed_everything(0)  # Fix the seed for reproducibility
    criterion = nn.L1Loss()
    TVLoss = TVL1(TVLoss_weight=1)
    # Load the parameters
    train_bs, val_bs, lr_DeepSR, lr_DeepAlign, num_epochs, num_im, sr_ratio, train_patchsize, val_patchsize, warp_weight, TVflow_weight= \
        args.train_bs, args.val_bs, args.lr_DeepSR, args.lr_DeepAlign, args.num_epochs, args.num_im, args.sr_ratio, args.train_patchsize, \
        args.val_patchsize, args.warp_weight, args.TVflow_weight
    Ed_num_features, PCAF_num_features, num_blocks= args.Ed_num_features, args.PCAF_num_features, args.num_blocks

    sr_ratio = args.sr_ratio
    folder_name = 'self-supervised_multi-image_deepSR_time_{}'.format(
        f"{datetime.datetime.now():%m-%d-%H-%M-%S}")

    ################## load Models
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    cudnn.benchmark = True
    args.local_rank = local_rank
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://127.0.0.1:5456',
                                         world_size=args.world_size,
                                         rank=local_rank)
    torch.cuda.set_device(local_rank)
    train_bs = int(train_bs / args.world_size)
    val_bs = int(val_bs / args.world_size)
    deepSuperresolve = DeepSR(local_rank=local_rank, num_im=num_im,
                              Encoder_num_features=Ed_num_features, PCAF_num_features=PCAF_num_features,
                              sr_ratio=sr_ratio, img_size=(train_patchsize*sr_ratio, train_patchsize*sr_ratio)).cuda(local_rank)


    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    if local_rank == 0:
        torch.save(deepSuperresolve.state_dict(), checkpoint_path)

    torch.distributed.barrier()
    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    deepSuperresolve.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda', local_rank)))
    base_params = list(map(id, deepSuperresolve.MEAlign.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, deepSuperresolve.parameters())  # 不包含返回真，过滤假
    DeepSR_params = [
        {"params": logits_params, "lr": lr_DeepSR},
        {"params": deepSuperresolve.MEAlign.parameters(), "lr": lr_DeepAlign}
    ]

    optimizer_DeepSR = torch.optim.AdamW(DeepSR_params, weight_decay=args.weight_decay)
    schedulerDeepSR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_DeepSR, num_epochs, eta_min=1e-6)

    if args.Sync_BN:
        # time-consuming using SyncBatchNorm
        deepSuperresolve = torch.nn.SyncBatchNorm.convert_sync_batchnorm(deepSuperresolve).cuda(local_rank)

    deepSuperresolve = torch.nn.parallel.DistributedDataParallel(deepSuperresolve, device_ids=[local_rank],
                                                                 broadcast_buffers=False)
    checkpoint_path_SR = "/pretrain_CASR-DSAT.pth.tar"
    checkpoint = torch.load(checkpoint_path_SR, map_location=torch.device('cuda', local_rank))
    deepSuperresolve.load_state_dict(checkpoint['state_dict DeepSR'])


    Dataset_path = "/train_our_real_CASR/"
    train_data_set = BurstSRDataset(root=Dataset_path, split='train', burst_size=9, crop_sz=args.train_patchsize,
                                    center_crop=False, random_flip=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    nw = min([os.cpu_count(), train_bs if train_bs >= 1 else 0, 8])  # number of workers
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=train_bs,
                                               sampler=train_sampler,
                                               pin_memory=True,
                                               num_workers=nw)
    val_data_set = BurstSRDataset(root=Dataset_path, split='val', burst_size=9, crop_sz=args.val_patchsize,
                                  center_crop=False, random_flip=False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)
    nw = min([os.cpu_count(), val_bs if val_bs >= 1 else 0, 8])
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=val_bs,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw)

    checkpoint_dir = '/ours_real'
    result_dir_SR = '/ours_real'
    result_dir_LR = '/ours_real'
    blur_filter_SR = BlurLayer().cuda(local_rank)   # blur kernel for the real output (to produce a sharp output directly)
    starttime = time()
    ##################
    for epoch in range(num_epochs): # checkpoint['epoch'],
        train_sampler.set_epoch(epoch)  # 打乱样本
        TrainLoss = []
        ValLoss = []
        PSNRLoss = []

        if local_rank == 0:  # 在第一进程中执行打印
            print('__________________________________________________')
            print('Training epoch {0:3d}'.format(epoch))

        for i, data in enumerate(train_loader):
            """
            samplesLR : b, num_im, h, w
            flow: b, num_im, 2, h, w
            SR:  b, 1, 3*h, 3*w

            """

            samplesLR = data
            optimizer_DeepSR.zero_grad()
            b, num_im, h, w = samplesLR.shape
            samplesLR = samplesLR.float().cuda(local_rank)
            c = 8  # partially overlapping border
            SR, flow, warploss, warped = deepSuperresolve(samplesLR.clone(), True, c, 'Detail') # Reference LR image is set to 0 during training
            SR = blur_filter_SR(SR)
            tvloss = TVLoss(flow[..., c:-c, c:-c])  # motion estimation loss: tv loss
            SR_ds = BackWarping(SR.unsqueeze(1).repeat(1, num_im, 1, 1, 1).view(-1, 1, sr_ratio * h, sr_ratio * w),
                                    local_rank,
                                    flow.view(-1, 2, h, w),
                                    sr_ratio)  # To align the SR with the reference image before downsampling
            samplesLR = samplesLR.view(b * num_im, 1, h, w)
            N2Nloss = criterion(samplesLR[::num_im, ..., c:-c, c:-c], SR_ds[::num_im, ..., c:-c, c:-c])

            trainloss = N2Nloss + warp_weight * warploss + TVflow_weight * tvloss
            trainloss.backward()
            optimizer_DeepSR.step()
            reduce_trainloss = reduce_mean(trainloss, args.world_size)
            TrainLoss.append(reduce_trainloss.data.item())

        torch.cuda.synchronize(torch.device('cuda', local_rank))

        if local_rank == 0:
            print('Train')
            print('{:.5f}'.format(mean(TrainLoss)))

        deepSuperresolve.eval()
        with torch.no_grad():
            for k, data in enumerate(val_loader):
                samplesLR = data
                b, num_im, h, w = samplesLR.shape
                samplesLR = samplesLR.float().cuda(local_rank)
                c = 8
                SR, flow, warploss, warped = deepSuperresolve(samplesLR.clone(), False, c, 'Detail') # Reference LR image remains unchanged during testing
                SR = blur_filter_SR(SR)
                tvloss = TVLoss(flow[..., c:-c, c:-c])
                SR_ds = BackWarping(
                    SR.unsqueeze(1).repeat(1, num_im, 1, 1, 1).view(-1, 1, sr_ratio * h, sr_ratio * w), local_rank,
                    flow.view(-1, 2, h, w),
                    sr_ratio)  # To align the SR with the reference frame before downsampling
                samplesLR = samplesLR.view(b * num_im, 1, h, w)

                N2Nloss = criterion(samplesLR[::num_im, ..., c:-c, c:-c], SR_ds[::num_im, ..., c:-c, c:-c])
                valloss = N2Nloss + warp_weight * warploss + TVflow_weight * tvloss
                reduce_valloss = reduce_mean(valloss, args.world_size)
                ValLoss.append(reduce_valloss.data.item())

            torch.cuda.synchronize(torch.device('cuda', local_rank))
            if local_rank == 0:
                SR = SR[0, 0, ...]
                SR = SR.detach().cpu().numpy().squeeze()
                cv2.imwrite(os.path.join(result_dir_SR, "SR_{:03d}.png".format(epoch)), SR)
                m, _, h, w = samplesLR.shape
                b = int(m / num_im)
                samplesLR = samplesLR.view(b, num_im, h, w)
                warped = warped.view(b, num_im, h, w)
                for n in range(num_im):
                    LR = samplesLR[0, n, ...]
                    LR = LR.detach().cpu().numpy().squeeze()
                    cv2.imwrite(os.path.join(result_dir_LR, "LR_{:03d}_{:02d}.png".format(epoch, n)), LR)
                    warp = warped[0, n, ...]
                    warp = warp.detach().cpu().numpy().squeeze()
                    cv2.imwrite(os.path.join(result_dir_LR, "Warped_{:03d}_{:02d}.png".format(epoch, n)), warp)

        if local_rank == 0:
            print('Val')
            print('{:.5f}'.format(mean(ValLoss)))

        schedulerDeepSR.step()

        if local_rank == 0:
            print('#### Saving Models ... ####')
            print('#### Saving Models ... ####')
            state = {'epoch': epoch + 1,
                     'state_dict DeepSR': deepSuperresolve.state_dict(),
                     'optimizerDeepSR': optimizer_DeepSR.state_dict(),
                     'schedulerDeepSR': schedulerDeepSR.state_dict()}
            torch.save(state, os.path.join(checkpoint_dir, 'checkpoint_{}.pth.tar'.format(epoch)))

    if local_rank == 0:
        print('Execution time = {:.0f}s'.format(time() - starttime))
    if local_rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    torch.distributed.destroy_process_group()
    return


def main(args):

    torch.cuda.empty_cache()
    mp.spawn(train, nprocs=args.world_size, args=(args.world_size, args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-bst", "--train_bs", help="Batch size of train loader", type=int, default=12)
    parser.add_argument("-bsv", "--val_bs", help="Batch size of val loader", type=int, default=12)
    parser.add_argument("-ls", "--lr_DeepSR", help="Learning rate of DeepSR", type=float, default=1e-5)
    parser.add_argument("-la", "--lr_DeepAlign", help="Learning rate of DeepAlign", type=float, default=1e-5)
    parser.add_argument("-ne", "--num_epochs", help="Num_epochs", type=int, default=25)
    parser.add_argument("-ednf", "--Ed_num_features", help="Num of features for encoder", type=int, default=64)
    parser.add_argument("-pbffnf", "--PBFF_num_features", help="Num of features for PBFF", type=int, default=64)
    parser.add_argument("-nb", "--num_blocks", help="Number of residual blocks in encoder", type=int, default=4)
    parser.add_argument("-srr", "--sr_ratio", help="Super-resolution factor", type=int, default=3)
    parser.add_argument('-num', '--num_im', nargs='+', help="Number of image for camera array", default=9)
    parser.add_argument('-tps', '--train_patchsize', help="the size of crop for training", default=56)
    parser.add_argument('-vps', '--val_patchsize', help="the size of crop for val", default=56)
    parser.add_argument('-wz', '--world_size', default=4, type=int, help='number of distributed processes')
    parser.add_argument("-ww", "--warp_weight", help="Weight for the warping loss", type=float, default=3.0)
    parser.add_argument("-tvw", "--TVflow_weight", help="Weight for the TV flow loss", type=float, default=0.01)
    parser.add_argument('-wd', '--weight-decay', help='weight decay (default: 1e-6)', type=float, default=1e-6)
    parser.add_argument('-sbn', '--Sync_BN', help="Synchronization of BN across multi-gpus", default=True)

    args = parser.parse_args()

    main(args)