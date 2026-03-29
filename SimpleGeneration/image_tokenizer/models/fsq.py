"""
https://github.com/duchenzhuang/FSQ-pytorch/blob/main/quantizers/fsq.py
"""

import os
import sys
import warnings

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import math
from einops import rearrange, pack, unpack

import torch
import torch.nn as nn
import torch.nn.functional as F

from SimpleGeneration.image_tokenizer.models.vqgan import Encoder, Decoder

__all__ = [
    'FSQ_8_codebook_4K',
    'FSQ_8_codebook_16K',
    'FSQ_8_codebook_64K',
    'FSQ_16_codebook_4K',
    'FSQ_16_codebook_16K',
    'FSQ_16_codebook_64K',
]


class FiniteScalarQuantizer(nn.Module):

    def __init__(self, levels=[7, 5, 5, 5, 5]):
        super(FiniteScalarQuantizer, self).__init__()
        levels = torch.tensor(levels, dtype=torch.int32)
        basis = torch.cumprod(torch.cat(
            [torch.ones(1, dtype=torch.int32), levels[:-1]]),
                              dim=0,
                              dtype=torch.int32)

        self.register_buffer("levels", levels, persistent=False)
        self.register_buffer("basis", basis, persistent=False)

        self.codebook_size = math.prod(levels)
        self.register_buffer("codebook_used",
                             torch.zeros(self.codebook_size, dtype=torch.bool))

    def reset_codebook_used(self):
        with torch.no_grad():
            self.codebook_used.zero_()

    def pack_one(self, t, pattern):
        return pack([t], pattern)

    def unpack_one(self, t, ps, pattern):
        return unpack(t, ps, pattern)[0]

    def bound(self, z, eps=1e-3):
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self.levels - 1) * (1 - eps) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()

        return (z + shift).tanh() * half_l - offset

    def round_ste(self, z):
        """Round with straight through gradients."""
        zhat = z.round()

        return z + (zhat - z).detach()

    def quantize(self, z):
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = self.round_ste(self.bound(z))
        # Renormalize to [-1, 1].
        half_width = self.levels // 2

        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self.levels // 2

        return (zhat_normalized * half_width) + half_width

    def codes_to_indices(self, zhat):
        """Converts a `code` to an index in the codebook."""
        zhat = self._scale_and_shift(zhat)

        return (zhat * self.basis).sum(dim=-1).to(torch.int32)

    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        # input z:[1, d, 32, 32], b=1, d=len(levels), h=32, w=32
        # For levels=[8,8,8,6,5], d=5; For levels=[8,8,8,5,5,5], d=6
        z = rearrange(z, 'b d ... -> b ... d')
        # [1, 1024, 6],1024=(32, 32)
        z, ps = self.pack_one(z, 'b * d')

        # [1, 1024, 1, 6],特征维度6拆分为c=1(码本数量)和d=6(每个码本的维度)
        z = rearrange(z, 'b n (c d) -> b n c d', c=1)

        # [1, 1024, 1, 6]
        codes = self.quantize(z)
        # [1, 1024, 1],通过对最后一个维度(d=6)进行线性组合并求和,生成每个位置的索引
        indices = self.codes_to_indices(codes)

        # [1, 1024, 6],将c=1和d=6合并回特征维度
        codes = rearrange(codes, 'b n c d -> b n (c d)')

        # [1, 32, 32, 6],恢复原始空间维度(32, 32)
        codes = self.unpack_one(codes, ps, 'b * d')
        # [1, 6, 32, 32]
        codes = rearrange(codes, 'b ... d -> b d ...')

        # 处理索引:[1, 32, 32]
        indices = self.unpack_one(indices, ps, 'b * c')
        indices = rearrange(indices, '... 1 -> ...')

        unique_indices = torch.unique(torch.flatten(indices))
        # 标记这些索引为已使用
        self.codebook_used[unique_indices] = True
        codebook_usage = self.codebook_used.sum().item() / self.codebook_size
        codebook_usage = torch.tensor(codebook_usage,
                                      dtype=torch.float,
                                      device=self.codebook_used.device)

        return codes, codebook_usage, indices


class FSQModel(nn.Module):

    def __init__(self,
                 levels=[7, 5, 5, 5, 5],
                 encoder_planes_mult=[1, 2, 2, 4],
                 decoder_planes_mult=[1, 2, 2, 4],
                 z_planes=256,
                 dropout_prob=0.0,
                 use_gradient_checkpoint=False):
        super(FSQModel, self).__init__()
        self.embedding_planes = len(levels)

        self.encoder = Encoder(planes_mult=encoder_planes_mult,
                               z_planes=z_planes,
                               dropout_prob=dropout_prob,
                               use_gradient_checkpoint=use_gradient_checkpoint)
        self.decoder = Decoder(planes_mult=decoder_planes_mult,
                               z_planes=z_planes,
                               dropout_prob=dropout_prob,
                               use_gradient_checkpoint=use_gradient_checkpoint)

        self.quantize = FiniteScalarQuantizer(levels=levels)

        self.quant_conv = nn.Conv2d(z_planes,
                                    self.embedding_planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=True)
        self.post_quant_conv = nn.Conv2d(self.embedding_planes,
                                         z_planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         bias=True)

    def forward(self, input):
        quant_t, codebook_usage, id_t = self.encode(input)
        dec = self.decode(quant_t)

        return dec, codebook_usage, id_t

    def encode(self, x):
        # image_size [1, 3, 256, 256]
        # [1, 256, 32, 32]
        x = self.encoder(x)
        # [1, 6, 32, 32]
        x = self.quant_conv(x)

        # quant_t:[1, 6, 32, 32]
        # id_t:[1, 32, 32]
        quant_t, codebook_usage, id_t = self.quantize(x)

        return quant_t, codebook_usage, id_t

    def decode(self, x):
        # image_size [1, 6, 32, 32]
        # [1, 256, 32, 32]
        x = self.post_quant_conv(x)
        # [1, 3, 256, 256]
        x = self.decoder(x)

        return x


def _fsq(levels, encoder_planes_mult, decoder_planes_mult, **kwargs):
    model = FSQModel(levels=levels,
                     encoder_planes_mult=encoder_planes_mult,
                     decoder_planes_mult=decoder_planes_mult,
                     **kwargs)

    return model


def FSQ_8_codebook_4K(**kwargs):
    return _fsq(levels=[7, 5, 5, 5, 5],
                encoder_planes_mult=[1, 2, 2, 4],
                decoder_planes_mult=[1, 2, 2, 4],
                **kwargs)


def FSQ_8_codebook_16K(**kwargs):
    return _fsq(levels=[8, 8, 8, 6, 5],
                encoder_planes_mult=[1, 2, 2, 4],
                decoder_planes_mult=[1, 2, 2, 4],
                **kwargs)


def FSQ_8_codebook_64K(**kwargs):
    return _fsq(levels=[8, 8, 8, 5, 5, 5],
                encoder_planes_mult=[1, 2, 2, 4],
                decoder_planes_mult=[1, 2, 2, 4],
                **kwargs)


def FSQ_16_codebook_4K(**kwargs):
    return _fsq(levels=[7, 5, 5, 5, 5],
                encoder_planes_mult=[1, 1, 2, 2, 4],
                decoder_planes_mult=[1, 1, 2, 2, 4],
                **kwargs)


def FSQ_16_codebook_16K(**kwargs):
    return _fsq(levels=[8, 8, 8, 6, 5],
                encoder_planes_mult=[1, 1, 2, 2, 4],
                decoder_planes_mult=[1, 1, 2, 2, 4],
                **kwargs)


def FSQ_16_codebook_64K(**kwargs):
    return _fsq(levels=[8, 8, 8, 5, 5, 5],
                encoder_planes_mult=[1, 1, 2, 2, 4],
                decoder_planes_mult=[1, 1, 2, 2, 4],
                **kwargs)


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

    net = FSQ_8_codebook_4K()
    image_h, image_w = 256, 256
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'input':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    dec, codebook_usage, id_t = net(
        torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, macs: {macs}, params: {params},dec_shape: {dec.shape}'
    )

    net = FSQ_16_codebook_4K()
    image_h, image_w = 256, 256
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'input':
                                              torch.randn(
                                                  1, 3, image_h, image_w),
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    dec, codebook_usage, id_t = net(
        torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, macs: {macs}, params: {params},dec_shape: {dec.shape}'
    )
