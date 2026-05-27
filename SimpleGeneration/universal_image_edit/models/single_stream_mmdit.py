"""
Full Single-Stream MMDIT Model

核心技术选型推断依据与结论见 design_doc.md

与 double_stream_mmdit.py 的区别：
- 全部使用 SingleStreamBlock，无 DoubleStreamBlock
- txt 和 img 在输入阶段即拼接为统一序列，一起通过所有block处理
- 结构更简单、参数效率更高、前向更快
- 参考来源：Z-Image 纯单流 + Flux2 SingleStreamBlock 设计 + ERNIE-Image 8B 单流 DiT
"""
import math

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = (x * rrms).to(x_dtype) * self.weight

        return x


class QKNorm(nn.Module):

    def __init__(self, dim):
        super(QKNorm, self).__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)

        q = q.to(v)
        k = k.to(v)

        return q, k


class SiLUGatedActivation(nn.Module):

    def __init__(self):
        super(SiLUGatedActivation, self).__init__()
        self.act = nn.SiLU(inplace=False)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        x = self.act(x1) * x2

        return x


class MLPTimeStepEmbedder(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super(MLPTimeStepEmbedder, self).__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        x = self.out_layer(self.silu(self.in_layer(x)))

        return x


class Modulation(nn.Module):

    def __init__(self, dim):
        super(Modulation, self).__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, vec):
        out = F.silu(vec)
        out = self.lin(out)
        if out.ndim == 2:
            out = out[:, None, :]
        out = out.chunk(3, dim=-1)

        return out


def rope(pos, dim, theta):
    """计算单轴RoPE频率矩阵"""
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out),
         torch.sin(out),
         torch.cos(out)],
        dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)

    return out.float()


def apply_rope(xq, xk, freqs_cis):
    """应用RoPE到Q和K张量"""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = (freqs_cis[..., 0] * xq_[..., 0] +
              freqs_cis[..., 1] * xq_[..., 1])
    xk_out = (freqs_cis[..., 0] * xk_[..., 0] +
              freqs_cis[..., 1] * xk_[..., 1])

    return (xq_out.reshape(*xq.shape).type_as(xq),
            xk_out.reshape(*xk.shape).type_as(xk))


class EmbedND(nn.Module):

    def __init__(self, dim, theta, axes_dim):
        super(EmbedND, self).__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat([
            rope(ids[..., i], self.axes_dim[i], self.theta)
            for i in range(n_axes)
        ],
                        dim=-3)
        emb = emb.unsqueeze(1)

        return emb


def timestep_embedding(t, dim, max_period=10000.0, time_factor=1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) *
        torch.arange(0, half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)

    return embedding


class SingleStreamBlock(nn.Module):
    """
    Single-Stream MMDIT Block

    txt和img已拼接为统一序列,一个linear同时产生QKV和MLP输入
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super(SingleStreamBlock, self).__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        # 合并linear: QKV(3*hidden) + MLP_in(2*mlp_hidden)
        self.linear1 = nn.Linear(hidden_size,
                                 hidden_size * 3 + self.mlp_hidden_dim * 2,
                                 bias=False)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim,
                                 hidden_size,
                                 bias=False)

        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size,
                                     elementwise_affine=False,
                                     eps=1e-6)
        self.mlp_act = SiLUGatedActivation()
        self.mod = Modulation(hidden_size)

    def forward(self, x, pe, vec):
        mod_shift, mod_scale, mod_gate = self.mod(vec)

        # Pre-norm + modulation
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift

        # Split into QKV and MLP input
        qkv, mlp = torch.split(self.linear1(x_mod),
                               [3 * self.hidden_dim, self.mlp_hidden_dim * 2],
                               dim=-1)

        q, k, v = rearrange(qkv,
                            "B L (K H D) -> K B H L D",
                            K=3,
                            H=self.num_heads)
        q, k = self.norm(q, k, v)

        # Apply RoPE
        q, k = apply_rope(q, k, pe)

        # F.scaled_dot_product_attention
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = rearrange(attn, "B H L D -> B L (H D)")

        # Concat attn output with MLP activation, project back
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))
        x = x + mod_gate * output

        return x


class LastLayer(nn.Module):

    def __init__(self, hidden_size, out_channels):
        super(LastLayer, self).__init__()
        self.norm_final = nn.LayerNorm(hidden_size,
                                       elementwise_affine=False,
                                       eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x, vec):
        mod = self.adaLN_modulation(vec)
        shift, scale = mod.chunk(2, dim=-1)
        if shift.ndim == 2:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        x = (1 + scale) * self.norm_final(x) + shift
        x = self.linear(x)

        return x


class FullSingleStreamMMDITModel(nn.Module):
    """
    支持：
    - 文生图（text-to-image）：仅输入文本
    - 图像编辑（image editing）：输入文本 + 1~N张参考图

    架构：全 Single-Stream Blocks → Final Layer
    条件：VLM embedding (frozen Qwen3.5-4B) + Timestep embedding
    """

    def __init__(self,
                 in_channels=128,
                 hidden_size=3072,
                 num_heads=24,
                 depth=40,
                 mlp_ratio=3.0,
                 axes_dim=[64, 64],
                 theta=2000.0,
                 context_in_dim=2560,
                 time_embed_dim=256,
                 use_gradient_checkpoint=False):
        super(FullSingleStreamMMDITModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.time_embed_dim = time_embed_dim
        self.use_gradient_checkpoint = use_gradient_checkpoint

        pe_dim = hidden_size // num_heads
        assert sum(axes_dim) == pe_dim, (f"axes_dim sum {sum(axes_dim)} != "
                                         f"head_dim {pe_dim}")

        # Position embedding
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)

        # Input projections
        self.img_in = nn.Linear(in_channels, hidden_size, bias=False)
        self.txt_in = nn.Linear(context_in_dim, hidden_size, bias=False)

        # Timestep embedding
        self.time_in = MLPTimeStepEmbedder(in_dim=time_embed_dim,
                                           hidden_dim=hidden_size)

        # Single-stream blocks
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = LastLayer(hidden_size, self.out_channels)

    def forward(self, x, x_ids, timesteps, ctx, ctx_ids):
        """
        Args:
            x: 图像latent tokens [B, N_img, C_in]
               文生图时 N_img = h*w (纯噪声)
               编辑时 N_img = h*w + N_ref (噪声 + 参考图latent)
            x_ids: 图像位置ID [B, N_img, 2]
            timesteps: 时间步 [B] (0~1范围)
            ctx: VLM文本embedding [B, N_txt, C_ctx]
            ctx_ids: 文本位置ID [B, N_txt, 2]

        Returns:
            velocity prediction [B, N_target, C_in]
            (仅返回前h*w个token的预测，不含参考图)
        """
        num_txt_tokens = ctx.shape[1]

        # Timestep embedding → conditioning vector
        t_emb = timestep_embedding(timesteps, self.time_embed_dim)
        vec = self.time_in(t_emb)

        # Project inputs
        img = self.img_in(x)
        txt = self.txt_in(ctx)

        # Position embeddings
        pe_img = self.pe_embedder(x_ids)
        pe_txt = self.pe_embedder(ctx_ids)

        x = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_txt, pe_img), dim=2)

        # Single-stream blocks
        for block in self.single_blocks:
            if self.use_gradient_checkpoint:
                x = checkpoint(block, x, pe, vec, use_reentrant=False)
            else:
                x = block(x, pe, vec)

        # Strip text tokens
        x = x[:, num_txt_tokens:, ...]

        # Final projection
        x = self.final_layer(x, vec)

        return x


# 推断依据：
# 1. 参数量估算公式（全单流 Transformer, SiLU-Gated FFN, bias=False）：
#    每个 SingleStreamBlock 参数：
#    - Modulation: Linear(d, 3d) → 3d²
#    - linear1: Linear(d, 3d + 2rd) → (3+2r)d²
#    - linear2: Linear(d + rd, d) → (1+r)d²
#    - QKNorm: 2 × head_dim (可忽略)
#    - 总计: (7 + 3r) × d²
#    mlp_ratio=3.0 时：每个 SingleStreamBlock ≈ 16d²
#    mlp_ratio=4.0 时：每个 SingleStreamBlock ≈ 19d²
#
#    其他参数（img_in + txt_in + time_in + final_layer）≈ 3d² + 3072d
#    总参数 ≈ depth × block_per + other
#
# 2. 参考配置验证：
#    - Flux2 Klein-4B: hidden=3072, single_depth≈20 → 约20×16×3072²≈2.9B (仅单流部分)
#    - Z-Image: hidden=3840, depth=30 (纯单流) → 约30×16×3840²≈7.1B
#    - ERNIE-Image: 8B 单流 DiT (README声称)
#
# 3. 设计原则（每一档模型的宽度和深度均严格大于前一档）：
#    - 1B 模型：轻量级，head_dim=64，mlp_ratio=4.0保证MLP表达力
#    - 2B 模型：head_dim=128，mlp_ratio=3.0，更宽更深
#    - 4B 模型：中等规模，平衡效果与效率
#    - 6B 模型：中大规模，接近Z-Image参数量
#    - 8B 模型：大规模，追求最优效果
#
# 4. RoPE 选择：
#    - theta=2000（与Flux2一致，适配图像latent序列长度256~4096）
#    - 纯 2D 图像只需 h, w 两个轴
#    - axes_dim 之和 = head_dim = hidden_size / num_heads
#    - 1B: head_dim=64, axes=[32,32]
#    - 2B/4B/6B/8B: head_dim=128, axes=[64,64]
#
# 5. VLM context_in_dim：
#    - Qwen3.5-4B 的 hidden_size = 2560
#    - 所有五档模型统一使用 context_in_dim=2560
#
# 6. 全局 bias=False：
#    - 与Flux2一致，所有Linear层均无bias
#    - 减少参数量，训练更稳定


def mmdit_single_1b(**kwargs):
    """1B 全单流模型：hidden=1536, heads=24, head_dim=64, axes=[32,32], ratio=4.0
    精算参数量：~1.00B, 深度=22
    宽度=1536
    """
    return FullSingleStreamMMDITModel(in_channels=128,
                                      hidden_size=1536,
                                      num_heads=24,
                                      depth=22,
                                      mlp_ratio=4.0,
                                      axes_dim=[32, 32],
                                      theta=2000.0,
                                      context_in_dim=2560,
                                      time_embed_dim=256,
                                      **kwargs)


def mmdit_single_2b(**kwargs):
    """2B 全单流模型：hidden=2048, heads=16, head_dim=128, axes=[64,64], ratio=3.0
    精算参数量：~2.03B, 深度=30
    宽度=2048 > 1B(1536), 深度=30 > 1B(22)
    """
    return FullSingleStreamMMDITModel(in_channels=128,
                                      hidden_size=2048,
                                      num_heads=16,
                                      depth=30,
                                      mlp_ratio=3.0,
                                      axes_dim=[64, 64],
                                      theta=2000.0,
                                      context_in_dim=2560,
                                      time_embed_dim=256,
                                      **kwargs)


def mmdit_single_4b(**kwargs):
    """4B 全单流模型：hidden=2560, heads=20, head_dim=128, axes=[64,64], ratio=3.0
    精算参数量：~4.01B, 深度=38
    宽度=2560 > 2B(2048), 深度=38 > 2B(30)
    """
    return FullSingleStreamMMDITModel(in_channels=128,
                                      hidden_size=2560,
                                      num_heads=20,
                                      depth=38,
                                      mlp_ratio=3.0,
                                      axes_dim=[64, 64],
                                      theta=2000.0,
                                      context_in_dim=2560,
                                      time_embed_dim=256,
                                      **kwargs)


def mmdit_single_6b(**kwargs):
    """6B 全单流模型：hidden=3072, heads=24, head_dim=128, axes=[64,64], ratio=3.0
    精算参数量：~6.08B, 深度=40
    宽度=3072 > 4B(2560), 深度=40 > 4B(38)
    """
    return FullSingleStreamMMDITModel(in_channels=128,
                                      hidden_size=3072,
                                      num_heads=24,
                                      depth=40,
                                      mlp_ratio=3.0,
                                      axes_dim=[64, 64],
                                      theta=2000.0,
                                      context_in_dim=2560,
                                      time_embed_dim=256,
                                      **kwargs)


def mmdit_single_8b(**kwargs):
    """8B 全单流模型：hidden=3456, heads=27, head_dim=128, axes=[64,64], ratio=3.0
    精算参数量：~8.07B, 深度=42
    宽度=3456 > 6B(3072), 深度=42 > 6B(40)
    """
    return FullSingleStreamMMDITModel(in_channels=128,
                                      hidden_size=3456,
                                      num_heads=27,
                                      depth=42,
                                      mlp_ratio=3.0,
                                      axes_dim=[64, 64],
                                      theta=2000.0,
                                      context_in_dim=2560,
                                      time_embed_dim=256,
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

    from calflops import calculate_flops

    # ==================== mmdit_single_1b ====================
    net = mmdit_single_1b()
    net = net.cuda()
    image_h, image_w = 16, 16
    x = torch.randn(1, image_h * image_w, 128).cuda()
    h_ids = torch.arange(image_h).unsqueeze(1).expand(image_h,
                                                      image_w).reshape(-1)
    w_ids = torch.arange(image_w).unsqueeze(0).expand(image_h,
                                                      image_w).reshape(-1)
    x_ids = torch.stack([h_ids, w_ids], dim=-1).unsqueeze(0).float().cuda()
    timesteps = torch.rand(1).cuda()
    ctx = torch.randn(1, 32, 2560).cuda()
    ctx_ids = torch.zeros(1, 32, 2).float().cuda()
    ctx_ids[..., 0] = torch.arange(32).unsqueeze(0)
    print('1111', x.shape, x_ids.shape, timesteps.shape, ctx.shape,
          ctx_ids.shape)
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x': x,
                                              'x_ids': x_ids,
                                              'timesteps': timesteps,
                                              'ctx': ctx,
                                              'ctx_ids': ctx_ids,
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'2222, flops: {flops}, macs: {macs}, params: {params}')
    outputs = net(x, x_ids, timesteps, ctx, ctx_ids)
    print('3333', outputs.shape)

    # mmdit_single_1b with gradient checkpoint
    net = mmdit_single_1b(use_gradient_checkpoint=True)
    net = net.cuda()
    image_h, image_w = 16, 16
    x = torch.randn(1, image_h * image_w, 128).cuda()
    h_ids = torch.arange(image_h).unsqueeze(1).expand(image_h,
                                                      image_w).reshape(-1)
    w_ids = torch.arange(image_w).unsqueeze(0).expand(image_h,
                                                      image_w).reshape(-1)
    x_ids = torch.stack([h_ids, w_ids], dim=-1).unsqueeze(0).float().cuda()
    timesteps = torch.rand(1).cuda()
    ctx = torch.randn(1, 32, 2560).cuda()
    ctx_ids = torch.zeros(1, 32, 2).float().cuda()
    ctx_ids[..., 0] = torch.arange(32).unsqueeze(0)
    print('1111', x.shape, x_ids.shape, timesteps.shape, ctx.shape,
          ctx_ids.shape)
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x': x,
                                              'x_ids': x_ids,
                                              'timesteps': timesteps,
                                              'ctx': ctx,
                                              'ctx_ids': ctx_ids,
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'2222, flops: {flops}, macs: {macs}, params: {params}')
    outputs = net(x, x_ids, timesteps, ctx, ctx_ids)
    print('3333', outputs.shape)

    # ==================== mmdit_single_2b ====================
    net = mmdit_single_2b()
    net = net.cuda()
    image_h, image_w = 16, 16
    x = torch.randn(1, image_h * image_w, 128).cuda()
    h_ids = torch.arange(image_h).unsqueeze(1).expand(image_h,
                                                      image_w).reshape(-1)
    w_ids = torch.arange(image_w).unsqueeze(0).expand(image_h,
                                                      image_w).reshape(-1)
    x_ids = torch.stack([h_ids, w_ids], dim=-1).unsqueeze(0).float().cuda()
    timesteps = torch.rand(1).cuda()
    ctx = torch.randn(1, 32, 2560).cuda()
    ctx_ids = torch.zeros(1, 32, 2).float().cuda()
    ctx_ids[..., 0] = torch.arange(32).unsqueeze(0)
    print('1111', x.shape, x_ids.shape, timesteps.shape, ctx.shape,
          ctx_ids.shape)
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x': x,
                                              'x_ids': x_ids,
                                              'timesteps': timesteps,
                                              'ctx': ctx,
                                              'ctx_ids': ctx_ids,
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'2222, flops: {flops}, macs: {macs}, params: {params}')
    outputs = net(x, x_ids, timesteps, ctx, ctx_ids)
    print('3333', outputs.shape)

    # ==================== mmdit_single_4b ====================
    net = mmdit_single_4b()
    net = net.cuda()
    image_h, image_w = 16, 16
    x = torch.randn(1, image_h * image_w, 128).cuda()
    h_ids = torch.arange(image_h).unsqueeze(1).expand(image_h,
                                                      image_w).reshape(-1)
    w_ids = torch.arange(image_w).unsqueeze(0).expand(image_h,
                                                      image_w).reshape(-1)
    x_ids = torch.stack([h_ids, w_ids], dim=-1).unsqueeze(0).float().cuda()
    timesteps = torch.rand(1).cuda()
    ctx = torch.randn(1, 32, 2560).cuda()
    ctx_ids = torch.zeros(1, 32, 2).float().cuda()
    ctx_ids[..., 0] = torch.arange(32).unsqueeze(0)
    print('1111', x.shape, x_ids.shape, timesteps.shape, ctx.shape,
          ctx_ids.shape)
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x': x,
                                              'x_ids': x_ids,
                                              'timesteps': timesteps,
                                              'ctx': ctx,
                                              'ctx_ids': ctx_ids,
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'2222, flops: {flops}, macs: {macs}, params: {params}')
    outputs = net(x, x_ids, timesteps, ctx, ctx_ids)
    print('3333', outputs.shape)

    # ==================== mmdit_single_6b ====================
    net = mmdit_single_6b()
    net = net.cuda()
    image_h, image_w = 16, 16
    x = torch.randn(1, image_h * image_w, 128).cuda()
    h_ids = torch.arange(image_h).unsqueeze(1).expand(image_h,
                                                      image_w).reshape(-1)
    w_ids = torch.arange(image_w).unsqueeze(0).expand(image_h,
                                                      image_w).reshape(-1)
    x_ids = torch.stack([h_ids, w_ids], dim=-1).unsqueeze(0).float().cuda()
    timesteps = torch.rand(1).cuda()
    ctx = torch.randn(1, 32, 2560).cuda()
    ctx_ids = torch.zeros(1, 32, 2).float().cuda()
    ctx_ids[..., 0] = torch.arange(32).unsqueeze(0)
    print('1111', x.shape, x_ids.shape, timesteps.shape, ctx.shape,
          ctx_ids.shape)
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x': x,
                                              'x_ids': x_ids,
                                              'timesteps': timesteps,
                                              'ctx': ctx,
                                              'ctx_ids': ctx_ids,
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'2222, flops: {flops}, macs: {macs}, params: {params}')
    outputs = net(x, x_ids, timesteps, ctx, ctx_ids)
    print('3333', outputs.shape)

    # ==================== mmdit_single_8b ====================
    net = mmdit_single_8b()
    net = net.cuda()
    image_h, image_w = 16, 16
    x = torch.randn(1, image_h * image_w, 128).cuda()
    h_ids = torch.arange(image_h).unsqueeze(1).expand(image_h,
                                                      image_w).reshape(-1)
    w_ids = torch.arange(image_w).unsqueeze(0).expand(image_h,
                                                      image_w).reshape(-1)
    x_ids = torch.stack([h_ids, w_ids], dim=-1).unsqueeze(0).float().cuda()
    timesteps = torch.rand(1).cuda()
    ctx = torch.randn(1, 32, 2560).cuda()
    ctx_ids = torch.zeros(1, 32, 2).float().cuda()
    ctx_ids[..., 0] = torch.arange(32).unsqueeze(0)
    print('1111', x.shape, x_ids.shape, timesteps.shape, ctx.shape,
          ctx_ids.shape)
    flops, macs, params = calculate_flops(model=net,
                                          kwargs={
                                              'x': x,
                                              'x_ids': x_ids,
                                              'timesteps': timesteps,
                                              'ctx': ctx,
                                              'ctx_ids': ctx_ids,
                                          },
                                          output_as_string=True,
                                          output_precision=3,
                                          print_results=False,
                                          print_detailed=False)
    print(f'2222, flops: {flops}, macs: {macs}, params: {params}')
    outputs = net(x, x_ids, timesteps, ctx, ctx_ids)
    print('3333', outputs.shape)
