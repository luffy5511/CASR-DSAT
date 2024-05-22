import random
import torch 
import torch.nn as nn

import numpy as np 

from torch.autograd import Variable
from torchvision.transforms import GaussianBlur

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def base_detail_decomp(samples, gaussian_filter):
    #samplesLR: b, num_im, h, w
    b, num_im, h, w = samples.shape
    base   = gaussian_filter(samples)
    detail = samples - base
    return base, detail #b, num_im, h, w

k = np.load("/home/WXS/CYT/相机阵列深度超分辨CAshiftAlign_swim_3/blur_kernel.npy").squeeze()
size = k.shape[0]

class BlurLayer(nn.Module):
    def __init__(self):
        super(BlurLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(size//2), 
            nn.Conv2d(1, 1, size, stride=1, padding=0, bias=False, groups=1)
        )

        self.weights_init()
    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            f.required_grad = False

Gaussian_Filter = GaussianBlur(11, sigma=1).to(device)

class TVL1(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVL1,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[-2]
        w_x = x.size()[-1]
    
        count_h = self._tensor_size(x[...,1:,:])
        count_w = self._tensor_size(x[...,:,1:])        
    
        h_tv = torch.abs((x[...,1:,:]-x[...,:h_x-1,:])).sum()
        w_tv = torch.abs((x[...,:,1:]-x[...,:,:w_x-1])).sum()
        #print("h,w:", h_tv, w_tv)
        return self.TVLoss_weight*(h_tv/count_h+w_tv/count_w)/batch_size
        #return self.TVLoss_weight*(h_tv+w_tv)/batch_size
        
    def _tensor_size(self,t):
        return t.size()[-3]*t.size()[-2]*t.size()[-1]


class WarpedLoss(nn.Module):
    def __init__(self, p = 1, interpolation = 'bilinear'):
        super(WarpedLoss, self).__init__()
        if p == 1:
            self.criterion = nn.L1Loss(reduction='mean') #change to reduction = 'mean'
        if p == 2:
            self.criterion = nn.MSELoss(reduction='mean')
        self.interpolation = interpolation
    def cubic_interpolation(self, A, B, C, D, x):
        a,b,c,d = A.size()
        x = x.view(a,1,c,d)#.repeat(1,3,1,1)
        return B + 0.5*x*(C - A + x*(2.*A - 5.*B + 4.*C - D + x*(3.*(B - C) + D - A)))

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        '''if torch.sum(flo*flo) == 0:
            return x
        else:
            
            B, C, H, W = x.size()

            # mesh grid
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()
            grid = grid.to(device)
            #print(grid.shape)
            vgrid = Variable(grid) + flo.to(device)

            if self.interpolation == 'bilinear':
                # scale grid to [-1,1] 
                vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
                vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

                vgrid = vgrid.permute(0,2,3,1)        
                output = nn.functional.grid_sample(x, vgrid,align_corners = True)

            if self.interpolation == 'bicubicTorch':
                # scale grid to [-1,1] 
                vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
                vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

                vgrid = vgrid.permute(0,2,3,1)        
                output = nn.functional.grid_sample(x, vgrid,align_corners = True,mode = 'bicubic')

            out_of_range_pixels = (vgrid <= -1) | (vgrid >= 1)
            out_of_range = out_of_range_pixels[..., 0] | out_of_range_pixels[..., 1]
            mask = torch.where(out_of_range.unsqueeze(1) == True, torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
                
                #mask = torch.ones(x.size()).cuda()
                #mask = nn.functional.grid_sample(mask, vgrid,align_corners = True,mode = 'bicubic')

                #mask[mask < 0.9999] = 0
                #mask[mask > 0] = 1
            return output, mask'''
        if torch.sum(flo * flo) == 0:
            return x
        else:

            B, C, H, W = x.size()

            # mesh grid
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()
            grid = grid.to(device)
            # print(grid.shape)
            vgrid = Variable(grid) + flo.to(device)

            if self.interpolation == 'bilinear':
                # scale grid to [-1,1]
                vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
                vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

                vgrid = vgrid.permute(0, 2, 3, 1)
                output = nn.functional.grid_sample(x, vgrid, align_corners=True)

            if self.interpolation == 'bicubicTorch':
                # scale grid to [-1,1]
                vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
                vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

                vgrid = vgrid.permute(0, 2, 3, 1)
                output = nn.functional.grid_sample(x, vgrid, align_corners=True, mode='bicubic')

                # mask = torch.ones(x.size()).cuda()
                # mask = nn.functional.grid_sample(mask, vgrid,align_corners = True,mode = 'bicubic')

                # mask[mask < 0.9999] = 0
                # mask[mask > 0] = 1
            return output  # , mask

    def forward(self, input, target, flow, c, losstype='L1', masks=None):
        # Warp input on target
        warped = self.warp(target, flow)

        input_ = input[..., c:-c, c:-c]
        warped_ = warped[..., c:-c, c:-c]
        if losstype == 'HighRes-net':
            warped_ = warped_ / torch.sum(warped_, dim=(2, 3), keepdim=True) * torch.sum(input_, dim=(2, 3),
                                                                                          keepdim=True)
        if losstype == 'Detail':
            _, warped_ = base_detail_decomp(warped_, Gaussian_Filter)
            _, input_ = base_detail_decomp(input_, Gaussian_Filter)

        if losstype == 'DetailReal':
            _, warped_ = base_detail_decomp(warped_, Gaussian_Filter)
            _, input_ = base_detail_decomp(input_, Gaussian_Filter)

            masks = masks[..., 2:-2, 2:-2]

            warped_ = warped_ * masks[:, :1] * masks[:, 1:]
            input_ = input_ * masks[:, :1] * masks[:, 1:]

        self.loss = self.criterion(input_, warped_)

        return self.loss, warped

    '''def forward(self, input, target, target_ori, flow, c, losstype='L1', masks=None):
        # Warp input on target
        warped, mask = self.warp(target, flow)
        warped_ori, _ = self.warp(target_ori, flow)

        if losstype == 'HighRes-net':
            warped_ = warped / torch.sum(mask*warped, dim=(2, 3), keepdim=True) * torch.sum(mask*input, dim=(2, 3), keepdim=True)
            input_ = input
        if losstype == 'Detail':
            _, warped_ = base_detail_decomp(warped, Gaussian_Filter)
            _, input_ = base_detail_decomp(input, Gaussian_Filter)

        if losstype == 'DetailReal':
            _, warped_ = base_detail_decomp(warped, Gaussian_Filter)
            _, input_ = base_detail_decomp(input, Gaussian_Filter)

            masks = masks[..., 2:-2, 2:-2]

            warped_ = warped_ * masks[:, :1] * masks[:, 1:]
            input_ = input_ * masks[:, :1] * masks[:, 1:]

        self.loss = self.criterion(mask * input_, mask * warped_)

        return self.loss, warped_ori, mask'''