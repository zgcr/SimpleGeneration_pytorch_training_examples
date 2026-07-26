"""
Z-Image AutoEncoder
Z-Image VAE uses the same architecture as FLUX.1 VAE.
https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py

Z-Image VAE config:
in_channels=3, out_channels=3, block_out_channels=[128, 256, 512, 512],
layers_per_block=2, latent_channels=16, scaling_factor=0.3611, shift_factor=0.1159,
use_quant_conv=False, use_post_quant_conv=False
"""
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint


class Upsample(nn.Module):

    def __init__(self, inplanes):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(inplanes,
                              inplanes,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x


class Downsample(nn.Module):

    def __init__(self, inplanes):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(inplanes,
                              inplanes,
                              kernel_size=3,
                              stride=2,
                              padding=0,
                              bias=True)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)

        return x


class ResnetBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super(ResnetBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.norm1 = nn.GroupNorm(num_groups=32,
                                  num_channels=inplanes,
                                  eps=1e-6)
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=planes, eps=1e-6)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)

        if self.inplanes != self.planes:
            self.nin_shortcut = nn.Conv2d(inplanes,
                                          planes,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bias=True)

        self.silu = nn.SiLU()

    def forward(self, x):
        h = x

        h = self.norm1(h)
        h = self.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.silu(h)
        h = self.conv2(h)

        if self.inplanes != self.planes:
            x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):

    def __init__(self, inplanes):
        super(AttnBlock, self).__init__()
        self.inplanes = inplanes

        self.norm = nn.GroupNorm(num_groups=32,
                                 num_channels=inplanes,
                                 eps=1e-6)

        self.q = nn.Conv2d(inplanes,
                           inplanes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           bias=True)
        self.k = nn.Conv2d(inplanes,
                           inplanes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           bias=True)
        self.v = nn.Conv2d(inplanes,
                           inplanes,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           bias=True)
        self.proj_out = nn.Conv2d(inplanes,
                                  inplanes,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias=True)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        batch, channel, height, width = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h,
                      "b 1 (h w) c -> b c h w",
                      h=height,
                      w=width,
                      c=channel,
                      b=batch)
        h = self.proj_out(h)

        return x + h


class Encoder(nn.Module):

    def __init__(self,
                 inplanes=3,
                 planes=128,
                 planes_mult=[1, 2, 4, 4],
                 res_block_nums=2,
                 z_planes=16,
                 use_gradient_checkpoint=False):
        super(Encoder, self).__init__()
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.stage_num = len(planes_mult)
        self.res_block_nums = res_block_nums

        self.conv_in = nn.Conv2d(inplanes,
                                 planes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=True)

        self.down = nn.ModuleList()
        inplanes_mult = (1, ) + tuple(planes_mult)
        block_inplanes = planes
        for level_idx in range(self.stage_num):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_inplanes = planes * inplanes_mult[level_idx]
            block_outplanes = planes * planes_mult[level_idx]
            for _ in range(self.res_block_nums):
                block.append(
                    ResnetBlock(inplanes=block_inplanes,
                                planes=block_outplanes))
                block_inplanes = block_outplanes

            down = nn.Module()
            down.block = block
            down.attn = attn

            if level_idx != self.stage_num - 1:
                down.downsample = Downsample(block_inplanes)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(inplanes=block_inplanes,
                                       planes=block_inplanes)
        self.mid.attn_1 = AttnBlock(block_inplanes)
        self.mid.block_2 = ResnetBlock(inplanes=block_inplanes,
                                       planes=block_inplanes)

        self.norm_out = nn.GroupNorm(num_groups=32,
                                     num_channels=block_inplanes,
                                     eps=1e-6)
        self.conv_out = nn.Conv2d(block_inplanes,
                                  2 * z_planes,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=True)

        self.silu = nn.SiLU()

    def forward(self, x):
        hs = [self.conv_in(x)]

        # downsampling
        for level_idx in range(self.stage_num):
            for i_block in range(self.res_block_nums):
                if self.use_gradient_checkpoint:
                    h = checkpoint(self.down[level_idx].block[i_block],
                                   hs[-1],
                                   use_reentrant=False)
                else:
                    h = self.down[level_idx].block[i_block](hs[-1])

                if len(self.down[level_idx].attn) > 0:
                    if self.use_gradient_checkpoint:
                        h = checkpoint(self.down[level_idx].attn[i_block],
                                       h,
                                       use_reentrant=False)
                    else:
                        h = self.down[level_idx].attn[i_block](h)

                hs.append(h)

            if level_idx != self.stage_num - 1:
                if self.use_gradient_checkpoint:
                    h = checkpoint(self.down[level_idx].downsample,
                                   hs[-1],
                                   use_reentrant=False)
                else:
                    h = self.down[level_idx].downsample(hs[-1])

                hs.append(h)

        # middle
        h = hs[-1]

        if self.use_gradient_checkpoint:
            h = checkpoint(self.mid.block_1, h, use_reentrant=False)
        else:
            h = self.mid.block_1(h)

        if self.use_gradient_checkpoint:
            h = checkpoint(self.mid.attn_1, h, use_reentrant=False)
        else:
            h = self.mid.attn_1(h)

        if self.use_gradient_checkpoint:
            h = checkpoint(self.mid.block_2, h, use_reentrant=False)
        else:
            h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = self.silu(h)
        h = self.conv_out(h)

        return h


class Decoder(nn.Module):

    def __init__(self,
                 z_planes=16,
                 planes=128,
                 planes_mult=[1, 2, 4, 4],
                 res_block_nums=2,
                 out_planes=3,
                 use_gradient_checkpoint=False):
        super(Decoder, self).__init__()
        self.stage_num = len(planes_mult)
        self.res_block_nums = res_block_nums
        self.use_gradient_checkpoint = use_gradient_checkpoint

        block_inplanes = planes * planes_mult[self.stage_num - 1]

        self.conv_in = nn.Conv2d(z_planes,
                                 block_inplanes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=True)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(inplanes=block_inplanes,
                                       planes=block_inplanes)
        self.mid.attn_1 = AttnBlock(block_inplanes)
        self.mid.block_2 = ResnetBlock(inplanes=block_inplanes,
                                       planes=block_inplanes)

        self.up = nn.ModuleList()
        for level_idx in reversed(range(self.stage_num)):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_outplanes = planes * planes_mult[level_idx]
            for _ in range(self.res_block_nums + 1):
                block.append(
                    ResnetBlock(inplanes=block_inplanes,
                                planes=block_outplanes))
                block_inplanes = block_outplanes

            up = nn.Module()
            up.block = block
            up.attn = attn

            if level_idx != 0:
                up.upsample = Upsample(block_inplanes)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(num_groups=32,
                                     num_channels=block_inplanes,
                                     eps=1e-6)
        self.conv_out = nn.Conv2d(block_inplanes,
                                  out_planes,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=True)

        self.silu = nn.SiLU()

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        if self.use_gradient_checkpoint:
            h = checkpoint(self.mid.block_1, h, use_reentrant=False)
        else:
            h = self.mid.block_1(h)

        if self.use_gradient_checkpoint:
            h = checkpoint(self.mid.attn_1, h, use_reentrant=False)
        else:
            h = self.mid.attn_1(h)

        if self.use_gradient_checkpoint:
            h = checkpoint(self.mid.block_2, h, use_reentrant=False)
        else:
            h = self.mid.block_2(h)

        # upsampling
        for level_idx in reversed(range(self.stage_num)):
            for i_block in range(self.res_block_nums + 1):
                if self.use_gradient_checkpoint:
                    h = checkpoint(self.up[level_idx].block[i_block],
                                   h,
                                   use_reentrant=False)
                else:
                    h = self.up[level_idx].block[i_block](h)

                if len(self.up[level_idx].attn) > 0:
                    if self.use_gradient_checkpoint:
                        h = checkpoint(self.up[level_idx].attn[i_block],
                                       h,
                                       use_reentrant=False)
                    else:
                        h = self.up[level_idx].attn[i_block](h)

            if level_idx != 0:
                if self.use_gradient_checkpoint:
                    h = checkpoint(self.up[level_idx].upsample,
                                   h,
                                   use_reentrant=False)
                else:
                    h = self.up[level_idx].upsample(h)

        # end
        h = self.norm_out(h)
        h = self.silu(h)
        h = self.conv_out(h)

        return h


class DiagonalGaussian(nn.Module):

    def __init__(self, sample=False, chunk_dim=1):
        super(DiagonalGaussian, self).__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z):
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)

        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean

    def kl(self, z):
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        logvar = torch.clamp(logvar, -30.0, 20.0)

        var = torch.exp(logvar)

        if self.sample:
            std = torch.exp(0.5 * logvar)
            z_out = mean + std * torch.randn_like(mean)
        else:
            z_out = mean

        kl_out = 0.5 * torch.sum(mean.pow(2) + var - 1.0 - logvar,
                                 dim=[1, 2, 3])

        return z_out, kl_out


# Z-Image VAE (same architecture as FLUX.1-dev VAE)
# inplanes=3,
# planes=128,
# planes_mult=[1, 2, 4, 4],
# res_block_nums=2,
# z_planes=16,
# out_planes=3,
# scale_factor=0.3611,
# shift_factor=0.1159,


class AutoEncoder(nn.Module):

    def __init__(self,
                 inplanes=3,
                 planes=128,
                 planes_mult=[1, 2, 4, 4],
                 res_block_nums=2,
                 z_planes=16,
                 out_planes=3,
                 scale_factor=0.3611,
                 shift_factor=0.1159,
                 logvar_init=0.0,
                 sample_z=False,
                 use_gradient_checkpoint=False):
        super(AutoEncoder, self).__init__()
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

        self.encoder = Encoder(inplanes=inplanes,
                               planes=planes,
                               planes_mult=planes_mult,
                               res_block_nums=res_block_nums,
                               z_planes=z_planes,
                               use_gradient_checkpoint=use_gradient_checkpoint)

        self.decoder = Decoder(z_planes=z_planes,
                               planes=planes,
                               planes_mult=planes_mult,
                               res_block_nums=res_block_nums,
                               out_planes=out_planes,
                               use_gradient_checkpoint=use_gradient_checkpoint)

        self.reg = DiagonalGaussian(sample=sample_z)

        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def encode(self, x):
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)

        return z

    def decode(self, z):
        z = z / self.scale_factor + self.shift_factor
        dec = self.decoder(z)

        return dec

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)

        return x

    def forward_train(self, x):
        x = self.encoder(x)
        z, kl_out = self.reg.kl(x)
        dec = self.decoder(z)

        return dec, kl_out

    def get_last_layer(self):
        return self.decoder.conv_out.weight


if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    net = AutoEncoder(inplanes=3,
                      planes=128,
                      planes_mult=[1, 2, 4, 4],
                      res_block_nums=2,
                      z_planes=16,
                      out_planes=3,
                      scale_factor=0.3611,
                      shift_factor=0.1159,
                      sample_z=False,
                      use_gradient_checkpoint=False)
    image_h, image_w = 256, 256
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    outs = net(torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, macs: {macs}, params: {params},out_shape: {outs.shape}'
    )

    net = AutoEncoder(inplanes=3,
                      planes=128,
                      planes_mult=[1, 2, 4, 4],
                      res_block_nums=2,
                      z_planes=16,
                      out_planes=3,
                      scale_factor=0.3611,
                      shift_factor=0.1159,
                      sample_z=False,
                      use_gradient_checkpoint=True)
    image_h, image_w = 256, 256
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    outs = net(torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, macs: {macs}, params: {params},out_shape: {outs.shape}'
    )
