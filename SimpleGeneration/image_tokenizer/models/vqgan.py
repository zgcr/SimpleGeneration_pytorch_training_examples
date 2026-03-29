"""
https://github.com/FoundationVision/LlamaGen/blob/main/tokenizer/tokenizer_image/vq_model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

__all__ = [
    'VQ_8_codebook_4096',
    'VQ_8_codebook_16384',
    'VQ_8_codebook_65536',
    'VQ_16_codebook_4096',
    'VQ_16_codebook_16384',
    'VQ_16_codebook_65536',
]


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

    def __init__(self, inplanes, planes, dropout_prob=0.):
        super(ResnetBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.norm1 = nn.GroupNorm(32, inplanes)
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)

        self.norm2 = nn.GroupNorm(32, planes)
        self.dropout = nn.Dropout(dropout_prob)
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
        h = self.dropout(h)
        h = self.conv2(h)

        if self.inplanes != self.planes:
            x = self.nin_shortcut(x)

        x = x + h

        return x


class AttnBlock(nn.Module):

    def __init__(self, inplanes):
        super(AttnBlock, self).__init__()
        self.norm = nn.GroupNorm(32, inplanes)

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
        h = x
        h = self.norm(h)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        batch, channel, height, width = q.shape

        # (batch, channel, height, width) -> (batch, height*width, channel)
        q = q.view(batch, channel, height * width).transpose(1, 2)
        k = k.view(batch, channel, height * width).transpose(1, 2)
        v = v.view(batch, channel, height * width).transpose(1, 2)

        # Attention = softmax(Q @ K^T / sqrt(d), dim=-1) @ V
        h = F.scaled_dot_product_attention(q, k, v)

        # (batch, height*width, channel) -> (batch, channel, height, width)
        h = h.transpose(1, 2).view(batch, channel, height, width)

        h = self.proj_out(h)

        x = x + h

        return x


class Encoder(nn.Module):

    def __init__(self,
                 inplanes=3,
                 planes=128,
                 planes_mult=[1, 2, 2, 4],
                 res_block_nums=2,
                 z_planes=256,
                 dropout_prob=0.0,
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

        # downsample
        self.conv_blocks = nn.ModuleList()
        inplanes_mult = (1, ) + tuple(planes_mult)
        for level_idx in range(self.stage_num):
            conv_block = nn.Module()
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()

            block_inplanes = planes * inplanes_mult[level_idx]
            block_outplanes = planes * planes_mult[level_idx]
            for _ in range(self.res_block_nums):
                res_block.append(
                    ResnetBlock(inplanes=block_inplanes,
                                planes=block_outplanes,
                                dropout_prob=dropout_prob))
                block_inplanes = block_outplanes
                if level_idx == self.stage_num - 1:
                    attn_block.append(AttnBlock(inplanes=block_inplanes))
            conv_block.res = res_block
            conv_block.attn = attn_block

            if level_idx != self.stage_num - 1:
                conv_block.downsample = Downsample(inplanes=block_inplanes)
            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(
            ResnetBlock(inplanes=block_inplanes,
                        planes=block_inplanes,
                        dropout_prob=dropout_prob))
        self.mid.append(AttnBlock(inplanes=block_inplanes))
        self.mid.append(
            ResnetBlock(inplanes=block_inplanes,
                        planes=block_inplanes,
                        dropout_prob=dropout_prob))

        # end
        self.norm_out = nn.GroupNorm(32, block_inplanes)
        self.conv_out = nn.Conv2d(block_inplanes,
                                  z_planes,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=True)

        self.silu = nn.SiLU()

    def forward(self, x):
        h = self.conv_in(x)

        # downsampling
        for level_idx, block in enumerate(self.conv_blocks):
            for i_block in range(self.res_block_nums):
                if self.use_gradient_checkpoint:
                    h = checkpoint(block.res[i_block], h, use_reentrant=False)
                else:
                    h = block.res[i_block](h)

                if len(block.attn) > 0:
                    if self.use_gradient_checkpoint:
                        h = checkpoint(block.attn[i_block],
                                       h,
                                       use_reentrant=False)
                    else:
                        h = block.attn[i_block](h)

            if level_idx != self.stage_num - 1:
                if self.use_gradient_checkpoint:
                    h = checkpoint(block.downsample, h, use_reentrant=False)
                else:
                    h = block.downsample(h)

        # middle
        for mid_block in self.mid:
            if self.use_gradient_checkpoint:
                h = checkpoint(mid_block, h, use_reentrant=False)
            else:
                h = mid_block(h)

        # end
        h = self.norm_out(h)
        h = self.silu(h)
        h = self.conv_out(h)

        return h


class Decoder(nn.Module):

    def __init__(self,
                 z_planes=256,
                 planes=128,
                 planes_mult=[1, 2, 2, 4],
                 res_block_nums=2,
                 out_planes=3,
                 dropout_prob=0.0,
                 use_gradient_checkpoint=False):
        super(Decoder, self).__init__()
        self.stage_num = len(planes_mult)
        self.res_block_nums = res_block_nums
        self.use_gradient_checkpoint = use_gradient_checkpoint

        block_inplanes = planes * planes_mult[self.stage_num - 1]

        # z to block_in
        self.conv_in = nn.Conv2d(z_planes,
                                 block_inplanes,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=True)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(
            ResnetBlock(inplanes=block_inplanes,
                        planes=block_inplanes,
                        dropout_prob=dropout_prob))
        self.mid.append(AttnBlock(inplanes=block_inplanes))
        self.mid.append(
            ResnetBlock(inplanes=block_inplanes,
                        planes=block_inplanes,
                        dropout_prob=dropout_prob))

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for level_idx in reversed(range(self.stage_num)):
            conv_block = nn.Module()
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()

            block_outplanes = planes * planes_mult[level_idx]
            for _ in range(self.res_block_nums + 1):
                res_block.append(
                    ResnetBlock(inplanes=block_inplanes,
                                planes=block_outplanes,
                                dropout_prob=dropout_prob))
                block_inplanes = block_outplanes
                if level_idx == self.stage_num - 1:
                    attn_block.append(AttnBlock(inplanes=block_inplanes))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # upsample
            if level_idx != 0:
                conv_block.upsample = Upsample(inplanes=block_inplanes)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = nn.GroupNorm(32, block_inplanes)
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
        for mid_block in self.mid:
            if self.use_gradient_checkpoint:
                h = checkpoint(mid_block, h, use_reentrant=False)
            else:
                h = mid_block(h)

        # upsampling
        for level_idx, block in enumerate(self.conv_blocks):
            for i_block in range(self.res_block_nums + 1):
                if self.use_gradient_checkpoint:
                    h = checkpoint(block.res[i_block], h, use_reentrant=False)
                else:
                    h = block.res[i_block](h)

                if len(block.attn) > 0:
                    if self.use_gradient_checkpoint:
                        h = checkpoint(block.attn[i_block],
                                       h,
                                       use_reentrant=False)
                    else:
                        h = block.attn[i_block](h)

            if level_idx != self.stage_num - 1:
                if self.use_gradient_checkpoint:
                    h = checkpoint(block.upsample, h, use_reentrant=False)
                else:
                    h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = self.silu(h)
        h = self.conv_out(h)

        return h


class VectorQuantizer(nn.Module):

    def __init__(self,
                 codebook_size=4096,
                 codebook_embedding_planes=8,
                 vq_loss_ratio=1.0,
                 commit_loss_beta_ratio=0.25,
                 entropy_loss_ratio=0.0):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size
        self.codebook_embedding_planes = codebook_embedding_planes
        self.vq_loss_ratio = vq_loss_ratio
        self.commit_loss_beta_ratio = commit_loss_beta_ratio
        self.entropy_loss_ratio = entropy_loss_ratio

        self.embedding = nn.Embedding(codebook_size, codebook_embedding_planes)
        nn.init.uniform_(self.embedding.weight, -1. / codebook_size,
                         1. / codebook_size)

        self.register_buffer("codebook_used",
                             torch.zeros(self.codebook_size, dtype=torch.bool))

    def reset_codebook_used(self):
        with torch.no_grad():
            self.codebook_used.zero_()

    def forward(self, z):
        # [B,C,H,W] -> [B,H,W,C]
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.codebook_embedding_planes)

        z = F.normalize(z, p=2, dim=-1)
        z_flattened = F.normalize(z_flattened, p=2, dim=-1)
        embedding = F.normalize(self.embedding.weight, p=2, dim=-1)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + torch.sum(
            embedding**2,
            dim=1) - 2 * torch.einsum('bd,dn->bn', z_flattened,
                                      torch.einsum('n d -> d n', embedding))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = embedding[min_encoding_indices].view(z.shape)

        vq_loss = self.vq_loss_ratio * F.mse_loss(z_q, z.detach())
        commit_loss = self.commit_loss_beta_ratio * F.mse_loss(z_q.detach(), z)
        entropy_loss = self.entropy_loss_ratio * self.compute_entropy_loss(-d)

        unique_indices = torch.unique(torch.flatten(min_encoding_indices))
        # 标记这些索引为已使用
        self.codebook_used[unique_indices] = True
        codebook_usage = self.codebook_used.sum().item() / self.codebook_size
        codebook_usage = torch.tensor(codebook_usage,
                                      dtype=torch.float,
                                      device=self.codebook_used.device)

        z_q = z + (z_q - z).detach()
        # [B,H,W,C] -> [B,C,H,W]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        loss_dict = {
            'vq_loss': vq_loss,
            'commit_loss': commit_loss,
            'entropy_loss': entropy_loss,
        }

        return z_q, codebook_usage, loss_dict

    def encode_image(self, z):
        # [B,C,H,W] -> [B,H,W,C]
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.codebook_embedding_planes)

        z = F.normalize(z, p=2, dim=-1)
        z_flattened = F.normalize(z_flattened, p=2, dim=-1)
        embedding = F.normalize(self.embedding.weight, p=2, dim=-1)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + torch.sum(
            embedding**2,
            dim=1) - 2 * torch.einsum('bd,dn->bn', z_flattened,
                                      torch.einsum('n d -> d n', embedding))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = embedding[min_encoding_indices].view(z.shape)

        z_q = z + (z_q - z).detach()
        # [B,H,W,C] -> [B,C,H,W]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, min_encoding_indices

    def update_codebook_usage(self, min_encoding_indices):
        unique_indices = torch.unique(torch.flatten(min_encoding_indices))
        # 标记这些索引为已使用
        self.codebook_used[unique_indices] = True
        codebook_usage = self.codebook_used.sum().item() / self.codebook_size

        return codebook_usage

    def compute_entropy_loss(self, affinity, temperature=0.01):
        flat_affinity = affinity.reshape(-1, affinity.shape[-1]) / temperature
        probs = F.softmax(flat_affinity, dim=-1)
        log_probs = F.log_softmax(flat_affinity, dim=-1)
        avg_probs = torch.mean(probs, dim=0)
        avg_entropy = -(avg_probs * (avg_probs + 1e-5).log()).sum()
        sample_entropy = -(probs * log_probs).sum(1).mean()
        loss = sample_entropy - avg_entropy

        return loss

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        embeddings = F.normalize(self.embedding.weight, p=2, dim=-1)
        # (b*h*w, c)
        z_q = embeddings[indices]

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)

        return z_q


class VQModel(nn.Module):

    def __init__(self,
                 codebook_size=4096,
                 codebook_embedding_planes=8,
                 vq_loss_ratio=1.0,
                 commit_loss_beta_ratio=0.25,
                 entropy_loss_ratio=0.0,
                 encoder_planes_mult=[1, 2, 2, 4],
                 decoder_planes_mult=[1, 2, 2, 4],
                 z_planes=256,
                 dropout_prob=0.0,
                 use_gradient_checkpoint=False):
        super(VQModel, self).__init__()
        self.encoder = Encoder(planes_mult=encoder_planes_mult,
                               z_planes=z_planes,
                               dropout_prob=dropout_prob,
                               use_gradient_checkpoint=use_gradient_checkpoint)
        self.decoder = Decoder(planes_mult=decoder_planes_mult,
                               z_planes=z_planes,
                               dropout_prob=dropout_prob,
                               use_gradient_checkpoint=use_gradient_checkpoint)

        self.quantize = VectorQuantizer(
            codebook_size=codebook_size,
            codebook_embedding_planes=codebook_embedding_planes,
            vq_loss_ratio=vq_loss_ratio,
            commit_loss_beta_ratio=commit_loss_beta_ratio,
            entropy_loss_ratio=entropy_loss_ratio)

        self.quant_conv = nn.Conv2d(z_planes,
                                    codebook_embedding_planes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=True)
        self.post_quant_conv = nn.Conv2d(codebook_embedding_planes,
                                         z_planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         bias=True)

    def encode(self, x):
        # VQ-16 image_size 256
        # input x [1, 3, 256, 256]
        # [1, 256, 16, 16]
        x = self.encoder(x)
        # [1, 8, 16, 16]
        x = self.quant_conv(x)
        # [1, 8, 16, 16]
        quant, codebook_usage, codebook_loss_dict = self.quantize(x)

        return quant, codebook_usage, codebook_loss_dict

    def decode(self, x):
        # VQ-16 image_size 256
        # input x [1, 8, 16, 16]
        # [1, 256, 16, 16]
        x = self.post_quant_conv(x)
        # [1, 3, 256, 256]
        x = self.decoder(x)

        return x

    def forward(self, input):
        # VQ-16 image_size 256
        # input [1, 3, 256, 256]
        # quant [1, 8, 16, 16]
        quant, codebook_usage, codebook_loss_dict = self.encode(input)
        # dec [16, 3, 256, 256]
        dec = self.decode(quant)

        return dec, codebook_usage, codebook_loss_dict

    def encode_image(self, x):
        # VQ-16 image_size 256
        # input x [1, 3, 256, 256]
        # [1, 256, 16, 16]
        x = self.encoder(x)
        # [1, 8, 16, 16]
        x = self.quant_conv(x)
        # [1, 8, 16, 16]
        quant, indices = self.quantize.encode_image(x)

        return quant, indices

    def decode_code(self, code_b, shape=None, channel_first=True):
        # code_b [256]
        # quant_b [1, 8, 16, 16]
        quant_b = self.quantize.get_codebook_entry(code_b, shape,
                                                   channel_first)
        # dec [1, 3, 256, 256]
        dec = self.decode(quant_b)

        return dec


def _vqgan(codebook_size, encoder_planes_mult, decoder_planes_mult, **kwargs):
    model = VQModel(codebook_size=codebook_size,
                    encoder_planes_mult=encoder_planes_mult,
                    decoder_planes_mult=decoder_planes_mult,
                    **kwargs)

    return model


def VQ_8_codebook_4096(**kwargs):
    return _vqgan(codebook_size=4096,
                  encoder_planes_mult=[1, 2, 2, 4],
                  decoder_planes_mult=[1, 2, 2, 4],
                  **kwargs)


def VQ_8_codebook_16384(**kwargs):
    return _vqgan(codebook_size=16384,
                  encoder_planes_mult=[1, 2, 2, 4],
                  decoder_planes_mult=[1, 2, 2, 4],
                  **kwargs)


def VQ_8_codebook_65536(**kwargs):
    return _vqgan(codebook_size=65536,
                  encoder_planes_mult=[1, 2, 2, 4],
                  decoder_planes_mult=[1, 2, 2, 4],
                  **kwargs)


def VQ_16_codebook_4096(**kwargs):
    return _vqgan(codebook_size=4096,
                  encoder_planes_mult=[1, 1, 2, 2, 4],
                  decoder_planes_mult=[1, 1, 2, 2, 4],
                  **kwargs)


def VQ_16_codebook_16384(**kwargs):
    return _vqgan(codebook_size=16384,
                  encoder_planes_mult=[1, 1, 2, 2, 4],
                  decoder_planes_mult=[1, 1, 2, 2, 4],
                  **kwargs)


def VQ_16_codebook_65536(**kwargs):
    return _vqgan(codebook_size=65536,
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

    net = VQ_8_codebook_4096()
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
    dec, codebook_usage, codebook_loss_dict = net(
        torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, macs: {macs}, params: {params},dec_shape: {dec.shape}'
    )

    net = VQ_16_codebook_4096()
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
    dec, codebook_usage, codebook_loss_dict = net(
        torch.autograd.Variable(torch.randn(2, 3, image_h, image_w)))
    print(
        f'1111, flops: {flops}, macs: {macs}, params: {params},dec_shape: {dec.shape}'
    )
