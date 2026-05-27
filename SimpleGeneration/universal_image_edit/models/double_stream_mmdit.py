"""
Double-Stream MMDIT Model

核心技术选型推断依据与结论见 design_doc.md
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

    def __init__(self, dim, double):
        super(Modulation, self).__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=False)

    def forward(self, vec):
        out = F.silu(vec)
        out = self.lin(out)
        if out.ndim == 2:
            out = out[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)

        if self.is_double:
            result = out[:3], out[3:]
        else:
            result = out[:3], None

        return result


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


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)


class DoubleStreamBlock(nn.Module):
    """
    Double-Stream MMDIT Block

    img流和txt流各自独立调制/norm/QKV/MLP,在attention阶段将Q/K/V拼接做joint attention
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio):
        super(DoubleStreamBlock, self).__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # Image stream
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size,
                                      elementwise_affine=False,
                                      eps=1e-6)
        self.img_attn = SelfAttention(hidden_size, num_heads)
        self.img_norm2 = nn.LayerNorm(hidden_size,
                                      elementwise_affine=False,
                                      eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False),
            SiLUGatedActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False))

        # Text stream
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size,
                                      elementwise_affine=False,
                                      eps=1e-6)
        self.txt_attn = SelfAttention(hidden_size, num_heads)
        self.txt_norm2 = nn.LayerNorm(hidden_size,
                                      elementwise_affine=False,
                                      eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False),
            SiLUGatedActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False))

    def forward(self, img, txt, vec, pe_img, pe_txt):
        # Modulation
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        img_m1_shift, img_m1_scale, img_m1_gate = img_mod1
        img_m2_shift, img_m2_scale, img_m2_gate = img_mod2
        txt_m1_shift, txt_m1_scale, txt_m1_gate = txt_mod1
        txt_m2_shift, txt_m2_scale, txt_m2_gate = txt_mod2

        # Prepare image QKV
        img_modulated = (1 + img_m1_scale) * self.img_norm1(img) + img_m1_shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv,
                                        "B L (K H D) -> K B H L D",
                                        K=3,
                                        H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # Prepare text QKV
        txt_modulated = (1 + txt_m1_scale) * self.txt_norm1(txt) + txt_m1_shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv,
                                        "B L (K H D) -> K B H L D",
                                        K=3,
                                        H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # Joint attention: concat txt + img
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # Apply RoPE
        pe = torch.cat((pe_txt, pe_img), dim=2)
        q, k = apply_rope(q, k, pe)

        # F.scaled_dot_product_attention (auto flash attn with BF16)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = rearrange(attn, "B H L D -> B L (H D)")

        # Split back
        num_txt = txt.shape[1]
        txt_attn = attn[:, :num_txt]
        img_attn = attn[:, num_txt:]

        # Image residuals
        img = img + img_m1_gate * self.img_attn.proj(img_attn)
        img = img + img_m2_gate * self.img_mlp(
            (1 + img_m2_scale) * self.img_norm2(img) + img_m2_shift)

        # Text residuals
        txt = txt + txt_m1_gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_m2_gate * self.txt_mlp(
            (1 + txt_m2_scale) * self.txt_norm2(txt) + txt_m2_shift)

        return img, txt


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
        self.mod = Modulation(hidden_size, double=False)

    def forward(self, x, pe, vec):
        mod_out, _ = self.mod(vec)
        mod_shift, mod_scale, mod_gate = mod_out

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


class MMDITModel(nn.Module):
    """
    支持：
    - 文生图（text-to-image）：仅输入文本
    - 图像编辑（image editing）：输入文本 + 1~N张参考图

    架构：Double-Stream Blocks → Single-Stream Blocks → Final Layer
    条件：VLM embedding (frozen Qwen3.5-4B) + Timestep embedding
    """

    def __init__(self,
                 in_channels=128,
                 hidden_size=3072,
                 num_heads=24,
                 depth=8,
                 depth_single_blocks=32,
                 mlp_ratio=4.0,
                 axes_dim=[64, 64],
                 theta=2000.0,
                 context_in_dim=2560,
                 time_embed_dim=256,
                 use_gradient_checkpoint=False):
        super(MMDITModel, self).__init__()
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

        # Double-stream blocks
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Single-stream blocks
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth_single_blocks)
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

        # Double-stream blocks
        for block in self.double_blocks:
            if self.use_gradient_checkpoint:
                img, txt = checkpoint(block,
                                      img,
                                      txt,
                                      vec,
                                      pe_img,
                                      pe_txt,
                                      use_reentrant=False)
            else:
                img, txt = block(img, txt, vec, pe_img, pe_txt)

        # Merge txt and img for single-stream
        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_txt, pe_img), dim=2)

        # Single-stream blocks
        for block in self.single_blocks:
            if self.use_gradient_checkpoint:
                img = checkpoint(block, img, pe, vec, use_reentrant=False)
            else:
                img = block(img, pe, vec)

        # Strip text tokens
        img = img[:, num_txt_tokens:, ...]

        # Final projection
        img = self.final_layer(img, vec)

        return img


# 推断依据：
# 1. 参数量估算公式（Transformer-based DiT, SiLU-Gated FFN, bias=False）：
#    每个流的参数：Modulation(6d²) + QKV(3d²) + Proj(d²) + MLP_up(2rd²) + MLP_down(rd²) = (10+3r)d²
#    mlp_ratio=4.0 时：
#    - 每个 Double-Stream Block ≈ 44*d² (img+txt各: (10+12)d² = 22d², 合计44d²)
#    - 每个 Single-Stream Block ≈ 19*d² (mod:3d² + linear1:11d² + linear2:5d²)
#    mlp_ratio=3.0 时：
#    - 每个 Double-Stream Block ≈ 38*d² (img+txt各: (10+9)d² = 19d², 合计38d²)
#    - 每个 Single-Stream Block ≈ 16*d² (mod:3d² + linear1:9d² + linear2:4d²)
#    总参数 ≈ depth_double × DS_per_block + depth_single × SS_per_block + other
#
# 2. Flux2 参考配置（全系列 mlp_ratio=3.0, theta=2000, bias=False）：
#    - Flux2-dev:  hidden=6144, heads=48, double=8, single=48 → ~12B
#    - Klein-4B:   hidden=3072, heads=24, double=5, single=20 → ~4B
#    - Klein-9B:   hidden=4096, heads=32, double=8, single=24 → ~9B
#
# 3. Z-Image 参考：hidden=3840, heads=30, layers=30 (single-only) → ~6B
#
# 4. 设计原则（每一档模型的宽度和深度均严格大于前一档）：
#    - 1B 模型：轻量级，适合快速实验和边缘部署（保持mlp_ratio=4.0，head_dim=64较小）
#    - 2B 模型：小规模，适合资源受限场景（mlp_ratio=3.0，更宽更深）
#    - 4B 模型：中等规模，平衡效果与效率（比Klein-4B更深）
#    - 6B 模型：中大规模，接近Klein-9B深度
#    - 8B 模型：大规模，追求最优效果（最宽最深）
#
# 5. RoPE 选择：
#    - theta=2000（与Flux2一致，适配图像latent序列长度256~4096）
#    - 纯 2D 图像只需 h, w 两个轴
#    - axes_dim 之和 = head_dim = hidden_size / num_heads
#    - 1B: head_dim=64, axes=[32,32]
#    - 2B/4B/6B/8B: head_dim=128, axes=[64,64]
#
# 6. VLM context_in_dim：
#    - Qwen3.5-4B 的 hidden_size = 2560
#    - 所有五档模型统一使用 context_in_dim=2560
#
# 7. 全局 bias=False：
#    - 与Flux2一致，所有Linear层（QKV/MLP/Modulation/TimeEmbed/LastLayer）均无bias
#    - 减少参数量，训练更稳定


def mmdit_1b(**kwargs):
    """1B 模型：hidden=1536, heads=24, head_dim=64, axes=[32,32], ratio=4.0
    精算参数量：~1.14B, 总层数=20 (double=4, single=16)
    宽度=1536, 深度=20
    """
    return MMDITModel(in_channels=128,
                      hidden_size=1536,
                      num_heads=24,
                      depth=4,
                      depth_single_blocks=16,
                      mlp_ratio=4.0,
                      axes_dim=[32, 32],
                      theta=2000.0,
                      context_in_dim=2560,
                      time_embed_dim=256,
                      **kwargs)


def mmdit_2b(**kwargs):
    """2B 模型：hidden=2048, heads=16, head_dim=128, axes=[64,64], ratio=3.0
    精算参数量：~2.02B, 总层数=23 (double=5, single=18)
    宽度=2048 > 1B(1536), 深度=23 > 1B(20)
    """
    return MMDITModel(in_channels=128,
                      hidden_size=2048,
                      num_heads=16,
                      depth=5,
                      depth_single_blocks=18,
                      mlp_ratio=3.0,
                      axes_dim=[64, 64],
                      theta=2000.0,
                      context_in_dim=2560,
                      time_embed_dim=256,
                      **kwargs)


def mmdit_4b(**kwargs):
    """4B 模型：hidden=2560, heads=20, head_dim=128, axes=[64,64], ratio=3.0
    精算参数量：~4.04B, 总层数=30 (double=6, single=24)
    宽度=2560 > 2B(2048), 深度=30 > 2B(23)
    """
    return MMDITModel(in_channels=128,
                      hidden_size=2560,
                      num_heads=20,
                      depth=6,
                      depth_single_blocks=24,
                      mlp_ratio=3.0,
                      axes_dim=[64, 64],
                      theta=2000.0,
                      context_in_dim=2560,
                      time_embed_dim=256,
                      **kwargs)


def mmdit_6b(**kwargs):
    """6B 模型：hidden=3072, heads=24, head_dim=128, axes=[64,64], ratio=3.0
    精算参数量：~6.32B, 总层数=32 (double=7, single=25)
    宽度=3072 > 4B(2560), 深度=32 > 4B(30)
    """
    return MMDITModel(in_channels=128,
                      hidden_size=3072,
                      num_heads=24,
                      depth=7,
                      depth_single_blocks=25,
                      mlp_ratio=3.0,
                      axes_dim=[64, 64],
                      theta=2000.0,
                      context_in_dim=2560,
                      time_embed_dim=256,
                      **kwargs)


def mmdit_8b(**kwargs):
    """8B 模型：hidden=3456, heads=27, head_dim=128, axes=[64,64], ratio=3.0
    精算参数量：~8.65B, 总层数=34 (double=8, single=26)
    宽度=3456 > 6B(3072), 深度=34 > 6B(32)
    """
    return MMDITModel(in_channels=128,
                      hidden_size=3456,
                      num_heads=27,
                      depth=8,
                      depth_single_blocks=26,
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

    # ==================== mmdit_1b ====================
    net = mmdit_1b()
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

    # mmdit_1b with gradient checkpoint
    net = mmdit_1b(use_gradient_checkpoint=True)
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

    # ==================== mmdit_2b ====================
    net = mmdit_2b()
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

    # ==================== mmdit_4b ====================
    net = mmdit_4b()
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

    # ==================== mmdit_6b ====================
    net = mmdit_6b()
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

    # ==================== mmdit_8b ====================
    net = mmdit_8b()
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
