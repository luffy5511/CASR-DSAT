""" Python script to train option J """
import datetime
import random
import numpy as np
# import matplotlib.pyplot as plt
import os
import argparse
import tempfile
from time import time
from cv2 import imwrite
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
from real_dataset.burstsr_dataset import BurstSRDataset

from torchvision.transforms import GaussianBlur
import torch.multiprocessing as mp
# from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Process
from seed_everything import seed_everything


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


def std(img, window_size=7):
    assert window_size % 2 == 1
    pad = window_size // 2

    # calculate std on the mean image of the color channels
    img = torch.mean(img, dim=1, keepdim=True)
    N, C, H, W = img.shape
    img = nn.functional.pad(img, [pad] * 4, mode='reflect')
    img = nn.functional.unfold(img, kernel_size=window_size)
    img = img.view(N, C, window_size * window_size, H, W)
    img = img - torch.mean(img, dim=2, keepdim=True)
    img = img * img
    img = torch.mean(img, dim=2, keepdim=True)
    img = torch.sqrt(img)
    img = img.squeeze(2)
    return img

def Texture(input, lower=8, upper=15):
    N, C, H, W = input.shape
    ratio = input.new_ones((N, 1, H, W)) * 0.5
    ratio[input < lower] = torch.sigmoid((input - lower))[input < lower]
    ratio[input > upper] = torch.sigmoid((input - upper))[input > upper]
    ratio = ratio.detach()

    return ratio


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
    args.dist_backend = 'nccl'
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
                                  ffn_expansion_factor=1.5,  # Expansion coefficient in gate module
                                  bias=False,
                                  LayerNorm_type='WithBias')
        # EncoderNet: feature extraction backbone based on channel self-attention transformer block
        self.PBFF = PCAFNet(num_im, PCAF_num_features)
        # PCAFNet: pseudo camera array feature
        self.APGU = AGFNet(PCAF_num_features, 4)
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

    def forward(self, samplesLR, flag_train, m2i_train, c, losstype, random_shifts):
        flow, warploss, warped = flowEstimation(samplesLR, self.local_rank, self.MEAlign, c,
                                                WarpedLoss(interpolation='bicubicTorch'),
                                                gaussian_filter=GaussianBlur(11, sigma=1), losstype=losstype)
        b, num_im, h, w = samplesLR.shape
        samplesLR = samplesLR.view(-1, 1, h, w)  # b*(num_im),1, h, w

        if flag_train:
            if m2i_train:
                samplesLR[::num_im, ...] = torch.tensor(0.0).cuda(self.local_rank)
                random_shifts = torch.randint(low=0, high=self.sr_ratio,
                                              size=(b, 1, 2, 1, 1)) / self.sr_ratio  # Grid shifting
                flow = flow - random_shifts.cuda(self.local_rank)
            else:
                flow = flow - random_shifts.cuda(self.local_rank)

        features = self.encoder(samplesLR).view(-1, h, w)  # b * (num_im), num_features, h, w
        flowf = flow.contiguous().view(-1, 1, 2, h, w).repeat(1, self.Encoder_num_features, 1, 1, 1).view(-1, 2, h, w)  # b * num_im* num_features, 2, h, w
        dadd = featureAdd(features, flowf, sr_ratio=self.sr_ratio,
                          local_rank=self.local_rank)  # b * num_im * num_features, sr_ration*h, sr_ratio*w
        # featureAdd: Sub-pixel motion compensation without parameters
        dadd = dadd.view(b, num_im, self.Encoder_num_features, self.sr_ratio * h, self.sr_ratio * w)
        SR = self.APGU(self.PBFF(dadd)).squeeze(1)  # b, 1, PCAF_num_features, sr_ratio * h, sr_ratio * w
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



def train(local_rank, world_size, args):
    seed_everything(0)  # Fix the seed for reproducibility
    criterion = nn.L1Loss()
    TVLoss = TVL1(TVLoss_weight=1)
    # Load the parameters
    train_bs, val_bs, lr_DeepSR, lr_DeepAlign, num_epochs, num_im, sr_ratio, train_patchsize, val_patchsize, warp_weight, TVflow_weight, m2i_weight = \
        args.train_bs, args.val_bs, args.lr_DeepSR, args.lr_DeepAlign, args.num_epochs, args.num_im, args.sr_ratio, args.train_patchsize, \
        args.val_patchsize, args.warp_weight, args.TVflow_weight, args.m2i_weight
    Ed_num_features, PCAF_num_features, num_blocks = args.Ed_num_features, args.PCAF_num_features, args.num_blocks

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
    deepSuperresolve_m2i = DeepSR(local_rank=local_rank, num_im=num_im,
                              Encoder_num_features=Ed_num_features, PCAF_num_features=PCAF_num_features,
                              sr_ratio=sr_ratio, img_size=(train_patchsize, train_patchsize)).cuda(local_rank)
    deepSuperresolve_gm2m = DeepSR(local_rank=local_rank, num_im=num_im,
                                  Encoder_num_features=Ed_num_features, PCAF_num_features=PCAF_num_features,
                                  sr_ratio=sr_ratio, img_size=(train_patchsize, train_patchsize)).cuda(local_rank)

    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    if local_rank == 0:
        torch.save(deepSuperresolve_m2i.state_dict(), checkpoint_path)
        torch.save(deepSuperresolve_gm2m.state_dict(), checkpoint_path)

    torch.distributed.barrier()
    deepSuperresolve_m2i.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda', local_rank)))
    deepSuperresolve_gm2m.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda', local_rank)))

    base_params_gm2m = list(map(id, deepSuperresolve_gm2m.MEAlign.parameters()))
    logits_params_gm2m = filter(lambda p: id(p) not in base_params_gm2m, deepSuperresolve_gm2m.parameters())
    DeepSR_params_gm2m = [
        {"params": logits_params_gm2m, "lr": lr_DeepSR},
        {"params": deepSuperresolve_gm2m.MEAlign.parameters(), "lr": lr_DeepAlign}
    ]

    optimizer_DeepSR_gm2m = torch.optim.AdamW(DeepSR_params_gm2m, weight_decay=args.weight_decay)
    schedulerDeepSR_gm2m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_DeepSR_gm2m, num_epochs, eta_min=1e-6)

    if args.Sync_BN:
        # time-consuming using SyncBatchNorm
        deepSuperresolve_gm2m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(deepSuperresolve_gm2m).cuda(local_rank)

    deepSuperresolve_m2i = torch.nn.parallel.DistributedDataParallel(deepSuperresolve_m2i, device_ids=[local_rank],
                                                                 broadcast_buffers=False)  # find_unused_parameters=False
    deepSuperresolve_gm2m = torch.nn.parallel.DistributedDataParallel(deepSuperresolve_gm2m, device_ids=[local_rank],
                                                                     broadcast_buffers=False)
    # To avoid the interference of M2M on M2I training, directly import the model parameters from each M2I training epoch.
    for param in deepSuperresolve_m2i.parameters():
        param.requires_grad = False

    checkpoint_path_SR = "/Pre-CASR-DSAT-GM2M.pth.tar"
    checkpoint = torch.load(checkpoint_path_SR, map_location=torch.device('cuda', local_rank))
    deepSuperresolve_gm2m.load_state_dict(checkpoint['state_dict DeepSR'])

    ################## load datas
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

    checkpoint_dir = '/GM2M_real'
    result_dir_SR = '/GM2M_real'
    result_dir_LR = '/GM2M_real'
    gaussian_filter = GaussianBlur(11, sigma=1).cuda()
    blur_filter_SR = BlurLayer().cuda(local_rank)
    ##################
    starttime = time()
    ##################
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        TrainLoss_gm2m = []
        PSNRLoss = []
        # Import the model parameters from each epoch of M2I training (train_m2i.py)
        deepSuperresolve_m2i_path = os.path.join("/M2I_real/", 'checkpoint_{}.pth.tar'.format(epoch))
        checkpoint = torch.load(deepSuperresolve_m2i_path, map_location=torch.device('cuda', local_rank))
        deepSuperresolve_m2i.load_state_dict(checkpoint['state_dict DeepSR'])

        if local_rank == 0:
            print('__________________________________________________')
            print('Training epoch {0:3d}'.format(epoch))

        for i, data in enumerate(train_loader):
            """
            samplesLR : b, num_im, h, w
            flow: b, num_im, 2, h, w
            SR:  b, 1, 3*h, 3*w

            """
            optimizer_DeepSR_gm2m.zero_grad()

            samplesLR = data
            b, num_im, h, w = samplesLR.shape
            samplesLR = samplesLR.float().cuda(local_rank)
            c = 8  # partially overlapping border
            random_shifts = torch.zeros(b, 1, 2, 1, 1)

            with torch.no_grad():
                SR_m2i, flow_m2i, warploss_m2i, random_shifts = deepSuperresolve_m2i(samplesLR.clone(), True, True, c, 'Detail', random_shifts)
                SR_m2i_blur = blur_filter_SR(SR_m2i)

            SR_m2m, flow_m2m, warploss_m2m, random_shifts = deepSuperresolve_gm2m(samplesLR.clone(), True, False, c, 'Detail', random_shifts)
            tvloss_m2m = TVLoss(flow_m2m[..., c:-c, c:-c]) # motion estimation loss: tv loss

            samplesLR = samplesLR.view(b * num_im, 1, h, w)
            # Extract detail of images
            samplesLR_base, samplesLR_detail = base_detail_decomp(samplesLR, gaussian_filter)
            # Calculate texture confidence
            std_value = std(samplesLR_detail, window_size=3)
            confidence_texture = Texture(std_value)
            binary_noise_sr = F.interpolate(confidence_texture[::num_im, ...], scale_factor=sr_ratio, mode='bilinear')
            SR_m2m_blur = binary_noise_sr * blur_filter_SR(SR_m2m) + (1 - binary_noise_sr) * SR_m2m
            SR_ds_m2m = BackWarping(SR_m2m_blur.unsqueeze(1).repeat(1, num_im, 1, 1, 1).view(-1, 1, sr_ratio * h, sr_ratio * w),
                                    local_rank,
                                    flow_m2m.view(-1, 2, h, w),
                                    sr_ratio)  # To align the SR with the reference image before downsampling
            SR_ds_m2m_base, SR_ds_m2m_detail = base_detail_decomp(SR_ds_m2m, gaussian_filter)
            # M2M loss
            highloss_m2m = criterion(confidence_texture[..., c:-c, c:-c] * SR_ds_m2m_detail[..., c:-c, c:-c],
                                     confidence_texture[..., c:-c, c:-c] * samplesLR_detail[..., c:-c, c:-c])
            N2Nloss_m2m = criterion((1-confidence_texture[::num_im, ..., c:-c, c:-c])*SR_ds_m2m[::num_im, ..., c:-c, c:-c], (1-confidence_texture[::num_im, ..., c:-c, c:-c])*samplesLR[::num_im, ..., c:-c, c:-c])
            # guided loss of output image from m2i training model
            superviseloss = criterion(binary_noise_sr[..., 3:-3, 3:-3]*SR_m2m_blur[..., 3:-3, 3:-3], binary_noise_sr[..., 3:-3, 3:-3]*SR_m2i_blur.detach()[..., 3:-3, 3:-3])

            trainloss_gm2m = N2Nloss_m2m + highloss_m2m + m2i_weight * superviseloss + warp_weight * warploss_m2m + TVflow_weight * tvloss_m2m
            trainloss_gm2m.backward()
            optimizer_DeepSR_gm2m.step()
            reduce_trainloss_gm2m = reduce_mean(trainloss_gm2m, args.world_size)
            TrainLoss_gm2m.append(reduce_trainloss_gm2m.data.item())
        torch.cuda.synchronize(torch.device('cuda', local_rank))

        if epoch < 300:
            if local_rank == 0:
                print('Train_m2m')
                print('{:.5f}'.format(mean(TrainLoss_gm2m)))

        deepSuperresolve_gm2m.eval()
        with torch.no_grad():
            for k, data in enumerate(val_loader):
                samplesLR = data
                b, num_im, h, w = samplesLR.shape
                samplesLR = samplesLR.float().cuda(local_rank)
                c = 8
                random_shifts = torch.zeros(b, 1, 2, 1, 1)
                SR, flow, warploss_m2m, random_shifts = deepSuperresolve_gm2m(samplesLR.clone(), False, False, c, 'Detail', random_shifts)

            torch.cuda.synchronize(torch.device('cuda', local_rank))
            if local_rank == 0:
                SR = SR[0, 0, ...]
                SR = SR.detach().cpu().numpy().squeeze()
                imwrite(os.path.join(result_dir_SR, "SR_{:03d}.png".format(epoch)), SR)


        schedulerDeepSR_gm2m.step()

        if local_rank == 0:
            print('#### Saving Models ... ####')
            print('#### Saving Models ... ####')
            state_m2m = {'epoch': epoch + 1,
                         'state_dict DeepSR': deepSuperresolve_gm2m.state_dict(),
                         'optimizerDeepSR': optimizer_DeepSR_gm2m.state_dict(),
                         'schedulerDeepSR': schedulerDeepSR_gm2m.state_dict()}
            torch.save(state_m2m, os.path.join(checkpoint_dir, 'GM2M_checkpoint_{}.tar'.format(epoch)))
    if local_rank == 0:
        print('Execution time = {:.0f}s'.format(time() - starttime))
    if local_rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    torch.distributed.destroy_process_group()
    # writer.close()
    return

def main(args):
    """
    Given a configuration, trains Encoder, Decoder and fnet for Multi-Frame Super Resolution (MFSR), and saves best model.
    Args:
        config: dict, configuration file
    """
    torch.cuda.empty_cache()
    mp.spawn(train, nprocs=args.world_size, args=(args.world_size, args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-bst", "--train_bs", help="Batch size of train loader", type=int, default=12)
    parser.add_argument("-bsv", "--val_bs", help="Batch size of val loader", type=int, default=12)
    parser.add_argument("-ls", "--lr_DeepSR", help="Learning rate of DeepSR", type=float, default=1e-4)
    parser.add_argument("-la", "--lr_DeepAlign", help="Learning rate of DeepAlign", type=float, default=1e-5)
    parser.add_argument("-ne", "--num_epochs", help="Num_epochs", type=int, default=100)
    parser.add_argument("-ednf", "--Ed_num_features", help="Num of features for encoder", type=int, default=64)
    parser.add_argument("-PCAFnf", "--PCAF_num_features", help="Num of features for PBFF", type=int, default=64)
    parser.add_argument("-nb", "--num_blocks", help="Number of residual blocks in encoder", type=int, default=4)
    parser.add_argument("-srr", "--sr_ratio", help="Super-resolution factor", type=int, default=3)
    parser.add_argument('-num', '--num_im', nargs='+', help="Number of image for camera array", default=9)
    parser.add_argument('-sbn', '--Sync_BN', help="Synchronization of BN across multi-gpus", default=True)
    parser.add_argument('-tps', '--train_patchsize', help="the size of crop for training", default=56)
    parser.add_argument('-vps', '--val_patchsize', help="the size of crop for val", default=56)
    parser.add_argument('-wz', '--world_size', default=3, type=int, help='number of distributed processes')
    parser.add_argument("-ww", "--warp_weight", help="Weight for the warping loss", type=float, default=3.0)
    parser.add_argument("-tvw", "--TVflow_weight", help="Weight for the TV flow loss", type=float, default=0.01)
    parser.add_argument("-m2iw", "--m2i_weight", help="Weight for the m2i loss", type=float, default=1.5)
    parser.add_argument('-wd', '--weight-decay', help='weight decay (default: 1e-6)', type=float, default=1e-6)

    args = parser.parse_args()

    main(args)