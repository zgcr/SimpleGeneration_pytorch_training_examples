"""
Flow Matching 损失函数

===============================================================================
推断依据：
===============================================================================
1. Flow Matching 基本原理：
   - 前向过程：z_t = (1 - sigma_t) * x_0 + sigma_t * noise
   - 目标（velocity）：v = noise - x_0
   - 模型预测 v_pred，损失 = MSE(v_pred, v_target)

2. 各库实现对比：
   - FireRed: target = noise - latents, weighted MSE with threshold clipping
     + compute_loss_weighting_for_sd3 (sigma-based weighting)
   - JoyAI: FlowMatchDiscreteScheduler, Euler solver, v-prediction
   - Z-Image: FlowMatchEulerDiscreteScheduler, dynamic shifting

3. Sigma 采样策略：
   - FireRed: logit-normal distribution (weighting_scheme)
   - Z-Image/JoyAI: uniform sampling on [0, 1]
   - SD3论文：logit-normal 效果更好，但uniform更简单

4. Loss weighting：
   - SD3: w(sigma) = sigma^(-2) 或其他 scheme
   - 简化方案：uniform weighting 或 min-SNR-gamma

===============================================================================
结论：
===============================================================================
采用标准 Flow Matching loss：
- 前向：z_t = (1 - t) * x_0 + t * noise, t ∈ [0, 1]
- 目标：v = noise - x_0
- 损失：MSE(v_pred, v_target)，支持可选的 sigma weighting
- 时间步采样：支持 uniform 和 logit-normal 两种
"""
import math
import torch
import torch.nn.functional as F
from torch import Tensor


def sample_timesteps_uniform(batch_size: int, device: torch.device) -> Tensor:
    """均匀采样时间步 t ∈ (0, 1)

    推断依据：最简单的采样策略，JoyAI/Z-Image默认使用。
    结论：作为默认选项，简单有效。
    """
    return torch.rand(batch_size, device=device)


def sample_timesteps_logit_normal(batch_size: int,
                                  device: torch.device,
                                  mean: float = 0.0,
                                  std: float = 1.0) -> Tensor:
    """Logit-Normal 分布采样时间步

    推断依据：SD3论文和FireRed使用，在中间时间步附近采样更多，
    因为中间时间步的梯度信号更有价值。
    结论：推荐用于正式训练，效果优于uniform。
    """
    u = torch.randn(batch_size, device=device) * std + mean
    t = torch.sigmoid(u)
    # 避免 t=0 或 t=1
    t = t.clamp(1e-5, 1.0 - 1e-5)
    return t


def compute_flow_matching_loss(
    model_output: Tensor,
    noise: Tensor,
    clean_latents: Tensor,
    sigmas: Tensor,
    weighting_scheme: str = "none",
) -> Tensor:
    """
    计算 Flow Matching 损失

    推断依据：
    - FireRed: custom_mse_loss with threshold clipping + weighting
    - 标准FM: MSE(v_pred, v_target) with optional sigma weighting
    结论：提供标准FM loss + 可选weighting，不使用threshold clipping（简化设计）

    Args:
        model_output: 模型预测的velocity [B, N, C]
        noise: 采样的噪声 [B, N, C]
        clean_latents: 干净的latent [B, N, C]
        sigmas: 时间步对应的sigma [B, 1, 1] 或 [B]
        weighting_scheme: "none", "sigma_sqrt", "min_snr_5"

    Returns:
        标量loss
    """
    # Flow matching target: v = noise - x_0
    target = noise - clean_latents

    # Per-element MSE
    loss = F.mse_loss(model_output.float(), target.float(), reduction="none")

    # Apply weighting
    if weighting_scheme == "sigma_sqrt":
        # w(sigma) = 1 / sigma，对高噪声级别降权
        # 推断依据：SD3的weighting方案之一
        w = 1.0 / (sigmas + 1e-6)
        while w.ndim < loss.ndim:
            w = w.unsqueeze(-1)
        loss = loss * w
    elif weighting_scheme == "min_snr_5":
        # Min-SNR-gamma with gamma=5
        # 推断依据：改进的SNR加权，防止高噪声级别梯度爆炸
        snr = (1.0 - sigmas) / (sigmas + 1e-6)
        gamma = 5.0
        w = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
        while w.ndim < loss.ndim:
            w = w.unsqueeze(-1)
        loss = loss * w

    return loss.mean()


def add_noise_flow_matching(clean_latents: Tensor, noise: Tensor,
                            timesteps: Tensor) -> Tensor:
    """
    Flow Matching 前向加噪

    推断依据：
    - FireRed: noisy = (1 - sigma) * latents + sigma * noise
    - 标准FM: z_t = (1 - t) * x_0 + t * epsilon
    结论：标准线性插值

    Args:
        clean_latents: 干净latent [B, N, C]
        noise: 高斯噪声 [B, N, C]
        timesteps: 时间步 t ∈ [0,1], [B]

    Returns:
        加噪后的latent [B, N, C]
    """
    t = timesteps
    while t.ndim < clean_latents.ndim:
        t = t.unsqueeze(-1)
    noisy_latents = (1.0 - t) * clean_latents + t * noise
    return noisy_latents


def compute_shift(image_seq_len: int,
                  base_seq_len: int = 256,
                  max_seq_len: int = 4096,
                  base_shift: float = 0.5,
                  max_shift: float = 1.15) -> float:
    """
    根据图像序列长度计算 flow shift 系数

    推断依据：
    - FireRed/Z-Image: 线性插值计算mu，用于dynamic shifting
    - 更长序列需要更大的shift以保证采样质量
    结论：标准线性插值方案

    Args:
        image_seq_len: 图像token序列长度
        base_seq_len: 基准序列长度
        max_seq_len: 最大序列长度
        base_shift: 基准shift值
        max_shift: 最大shift值

    Returns:
        shift系数mu
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu
