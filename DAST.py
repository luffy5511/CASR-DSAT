## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

from seed_everything import seed_everything
import swin.swin_util as swu

seed_everything(0)
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


#########################residual TransformerGroup###############################
class ResTransformerGroup(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_block):
        super(ResTransformerGroup, self).__init__()
        self.TransformerGroup = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_block)])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = x + self.conv(self.TransformerGroup(x))
        return x
#########################residual SwinTransformerGroup###############################
class ResSwinTransformerGroup(nn.Module):
    def __init__(self, dim, num_heads, num_block, img_size, window_size=7, patch_size=1, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, bias=False, patch_norm=True, ape=False):
        super(ResSwinTransformerGroup, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_block)]
        # split image into non-overlapping patches
        self.patch_embed = swu.PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim,
            norm_layer=norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into imageape
        self.patch_unembed = swu.PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim,
            norm_layer=norm_layer if patch_norm else None)
        self.patches_resolution = self.patch_embed.patches_resolution

        self.layers = nn.ModuleList()
        for i_layer in range(num_block):
            layer = swu.SwinTransformerBlock(dim=dim,
                                             input_resolution=(patches_resolution[0],
                                                               patches_resolution[1]),
                                             num_heads=num_heads, window_size=window_size,
                                             shift_size=0 if (i_layer % 2 == 0) else window_size // 2,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop_rate, attn_drop=attn_drop_rate,
                                             drop_path=dpr[i_layer],
                                             norm_layer=norm_layer)
            self.layers.append(layer)
        self.norm = norm_layer(dim)

        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.ape = ape
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
            swu.trunc_normal_(self.absolute_pos_embed, std=.02)

    def forward(self, x):
        x_size = (x.shape[-2], x.shape[-1])
        x = self.patch_embed(x, use_norm=True)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for idx, layer in enumerate(self.layers):
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        x = x + self.conv(x)
        return x

###########################EncoderNet######################################
class EncoderNet(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=64,
                 num_blocks=[1, 2, 4],
                 heads=[1, 2, 4],
                 ffn_expansion_factor=2.66,  # 门控模块里的膨胀系数，为了增加通道数量
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(EncoderNet, self).__init__()
        self.inputConv = nn.Conv2d(in_channels=inp_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.encoder_level1 = ResTransformerGroup(dim=out_channels, num_heads=heads[0],
                                                  ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                  LayerNorm_type=LayerNorm_type, num_block=num_blocks[0])

        self.down1_2 = Downsample(out_channels)  ## From Level 1 to Level 2
        self.encoder_level2 = ResTransformerGroup(dim=int(out_channels * 2 ** 1), num_heads=heads[1],
                                                  ffn_expansion_factor=ffn_expansion_factor,
                                                  bias=bias, LayerNorm_type=LayerNorm_type, num_block=num_blocks[1])

        self.down2_3 = Downsample(int(out_channels * 2 ** 1))  ## From Level 2 to Level 3
        self.latent = ResTransformerGroup(dim=int(out_channels * 2 ** 2), num_heads=heads[2],
                                          ffn_expansion_factor=ffn_expansion_factor,
                                          bias=bias, LayerNorm_type=LayerNorm_type, num_block=num_blocks[2])

        self.up3_2 = Upsample(int(out_channels * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(out_channels * 2 ** 2), int(out_channels * 2 ** 1), kernel_size=1,
                                            bias=bias)
        self.decoder_level2 = ResTransformerGroup(dim=int(out_channels * 2 ** 1), num_heads=heads[1],
                                                  ffn_expansion_factor=ffn_expansion_factor,
                                                  bias=bias, LayerNorm_type=LayerNorm_type, num_block=num_blocks[1])

        self.up2_1 = Upsample(int(out_channels * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Conv2d(int(out_channels * 2 ** 1), int(out_channels), kernel_size=1,
                                            bias=bias)
        self.decoder_level1 = ResTransformerGroup(dim=int(out_channels), num_heads=heads[0],
                                                  ffn_expansion_factor=ffn_expansion_factor,
                                                  bias=bias, LayerNorm_type=LayerNorm_type, num_block=num_blocks[0])

    def forward(self, inp_img):
        out_conv = self.inputConv(inp_img)
        out_enc_level1 = self.encoder_level1(out_conv)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        latent = self.latent(inp_enc_level3)

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        return out_dec_level1



###########################DecoderNet######################################

class DecoderNet(nn.Module):
    def __init__(self,
                 inp_channels=64,
                 out_channels=1,
                 num_blocks=[1, 2, 4],
                 heads=[1, 2, 4],
                 img_size=(192, 192),
                 bias=False):  ## Other option 'BiasFree'

        super(DecoderNet, self).__init__()

        self.encoder_level1 = ResSwinTransformerGroup(dim=inp_channels, num_heads=heads[0],
                                                  img_size=img_size, patch_size=1, bias=bias, num_block=num_blocks[0])

        self.down1_2 = Downsample(inp_channels)  ## From Level 1 to Level 2
        self.encoder_level2 = ResSwinTransformerGroup(dim=int(inp_channels * 2 ** 1), num_heads=heads[1],
                                                  img_size=(img_size[0]/2, img_size[1]/2), patch_size=1, bias=bias, num_block=num_blocks[1])

        self.down2_3 = Downsample(int(inp_channels * 2 ** 1))  ## From Level 2 to Level 3
        self.latent = ResSwinTransformerGroup(dim=int(inp_channels * 2 ** 2), num_heads=heads[2],
                                          img_size=(img_size[0]/4, img_size[1]/4), patch_size=1, bias=bias, num_block=num_blocks[2])

        self.up3_2 = Upsample(int(inp_channels * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(inp_channels * 2 ** 2), int(inp_channels * 2 ** 1), kernel_size=1,
                                            bias=bias)
        self.decoder_level2 = ResSwinTransformerGroup(dim=int(inp_channels * 2 ** 1), num_heads=heads[1],
                                                  img_size=(img_size[0]/2, img_size[1]/2), patch_size=1, bias=bias, num_block=num_blocks[1])

        self.up2_1 = Upsample(int(inp_channels * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = ResSwinTransformerGroup(dim=int(inp_channels * 2 ** 1), num_heads=heads[0],
                                                  img_size=img_size, patch_size=1, bias=bias, num_block=num_blocks[0])

        self.output = nn.Conv2d(int(inp_channels * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        out_enc_level1 = self.encoder_level1(inp_img)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        latent = self.latent(inp_enc_level3)

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)  # 每个编码层和解码层都改成残差transformer块

        return out_dec_level1
# Transformer空间注意力,但是是作为编码器
class DecoderNet2(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=64,
                 num_blocks=[1, 2, 4],
                 heads=[1, 2, 4],
                 img_size=(192, 192),
                 bias=False):  ## Other option 'BiasFree'

        super(DecoderNet2, self).__init__()

        self.inputConv = nn.Conv2d(in_channels=inp_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                   padding=1, padding_mode='reflect')

        self.encoder_level1 = ResSwinTransformerGroup(dim=out_channels, num_heads=heads[0],
                                                  img_size=img_size, patch_size=1, bias=bias, num_block=num_blocks[0])

        self.down1_2 = Downsample(out_channels)  ## From Level 1 to Level 2
        self.encoder_level2 = ResSwinTransformerGroup(dim=int(out_channels * 2 ** 1), num_heads=heads[1],
                                                  img_size=(img_size[0]/2, img_size[1]/2), patch_size=1, bias=bias, num_block=num_blocks[1])

        self.down2_3 = Downsample(int(out_channels * 2 ** 1))  ## From Level 2 to Level 3
        self.latent = ResSwinTransformerGroup(dim=int(out_channels * 2 ** 2), num_heads=heads[2],
                                          img_size=(img_size[0]/4, img_size[1]/4), patch_size=1, bias=bias, num_block=num_blocks[2])

        self.up3_2 = Upsample(int(out_channels * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(out_channels * 2 ** 2), int(out_channels * 2 ** 1), kernel_size=1,
                                            bias=bias)
        self.decoder_level2 = ResSwinTransformerGroup(dim=int(out_channels * 2 ** 1), num_heads=heads[1],
                                                  img_size=(img_size[0]/2, img_size[1]/2), patch_size=1, bias=bias, num_block=num_blocks[1])

        self.up2_1 = Upsample(int(out_channels * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Conv2d(int(out_channels * 2 ** 1), int(out_channels), kernel_size=1,
                                            bias=bias)
        self.decoder_level1 = ResSwinTransformerGroup(dim=int(out_channels), num_heads=heads[0],
                                                  img_size=img_size, patch_size=1, bias=bias, num_block=num_blocks[0])

    def forward(self, inp_img):
        out_conv = self.inputConv(inp_img)
        out_enc_level1 = self.encoder_level1(out_conv)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        latent = self.latent(inp_enc_level3)

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        return out_dec_level1

'''
class TMENet(nn.Module):
    def __init__(self,
                 inp_channels=64,
                 out_channels=1,
                 num_blocks=[2, 2],
                 heads=[2, 2],
                 img_size=192,
                 bias=False):  ## Other option 'BiasFree'

        super(TMENet, self).__init__()

        self.encoder_level1 = ResSwinTransformerGroup(dim=inp_channels, num_heads=heads[0],
                                                  img_size=img_size, patch_size=1, bias=bias, num_block=num_blocks[0])

        self.down1_2 = Downsample(inp_channels)  ## From Level 1 to Level 2
        self.latent = ResSwinTransformerGroup(dim=int(inp_channels * 2 ** 1), num_heads=heads[1],
                                                  img_size=img_size/2, patch_size=2, bias=bias, num_block=num_blocks[1])

        self.up2_1 = Upsample(int(inp_channels * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = ResSwinTransformerGroup(dim=int(inp_channels * 2 ** 1), num_heads=heads[0],
                                                  img_size=img_size, patch_size=1, bias=bias, num_block=num_blocks[0])

        self.output = nn.Conv2d(int(inp_channels * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        out_enc_level1 = self.encoder_level1(inp_img)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        latent = self.latent(inp_enc_level2)

        inp_dec_level1 = self.up2_1(latent)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)  # 每个编码层和解码层都改成残差transformer块

        return out_dec_level1
'''
class TMENet(nn.Module):
    def __init__(self,
                 inp_channels=64,
                 out_channels=1,
                 num_blocks=[2, 2, 2],
                 heads=[2, 2, 2],
                 img_size=192,
                 bias=False):  ## Other option 'BiasFree'

        super(TMENet, self).__init__()

        self.encoder_level1 = ResSwinTransformerGroup(dim=inp_channels, num_heads=heads[0],
                                                      img_size=img_size, patch_size=1, bias=bias,
                                                      num_block=num_blocks[0])

        self.down1_2 = Downsample(inp_channels)  ## From Level 1 to Level 2
        self.encoder_level2 = ResSwinTransformerGroup(dim=int(inp_channels * 2 ** 1), num_heads=heads[1],
                                                      img_size=img_size / 2, patch_size=1, bias=bias,
                                                      num_block=num_blocks[1])

        self.down2_3 = Downsample(int(inp_channels * 2 ** 1))  ## From Level 2 to Level 3
        self.latent = ResSwinTransformerGroup(dim=int(inp_channels * 2 ** 2), num_heads=heads[2],
                                              img_size=img_size / 4, patch_size=1, bias=bias, num_block=num_blocks[2])

        self.up3_2 = Upsample(int(inp_channels * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(inp_channels * 2 ** 2), int(inp_channels * 2 ** 1), kernel_size=1,
                                            bias=bias)
        self.decoder_level2 = ResSwinTransformerGroup(dim=int(inp_channels * 2 ** 1), num_heads=heads[1],
                                                      img_size=img_size / 2, patch_size=1, bias=bias,
                                                      num_block=num_blocks[1])

        self.up2_1 = Upsample(int(inp_channels * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = ResSwinTransformerGroup(dim=int(inp_channels * 2 ** 1), num_heads=heads[0],
                                                      img_size=img_size, patch_size=1, bias=bias,
                                                      num_block=num_blocks[0])

        self.output = nn.Conv2d(int(inp_channels * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        out_enc_level1 = self.encoder_level1(inp_img)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        latent = self.latent(inp_enc_level3)

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)  # 每个编码层和解码层都改成残差transformer块

        return out_dec_level1