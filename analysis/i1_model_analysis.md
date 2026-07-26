# i1 模型全面代码分析报告

> 基于 `i1/` 目录下的完整代码实现进行分析  
> 论文：*i1: A Simple and Fully Open Recipe for Strong Text-to-Image Models*（Princeton University）

---

## 目录

1. [VAE 分析](#1-vae-分析)
2. [DIT 模型分析](#2-dit-模型分析)
3. [子网络结构](#3-子网络结构)
4. [模型网络结构图](#4-模型网络结构图)
5. [模型能力分析](#5-模型能力分析)
6. [文生图流程图](#6-文生图流程图)
7. [图像编辑流程图](#7-图像编辑流程图)
8. [相比 FLUX.2 模型的创新点](#8-相比-flux2-模型的创新点)

---

## 1. VAE 分析

### 问题：这个模型是否使用了VAE？若使用了VAE，使用的VAE结构是什么？

### 结论：**i1 使用了 VAE，具体使用的是 FLUX.2 模型的 VAE 结构。**

### 代码分析依据

#### 1.1 VAE 配置定义

在 `i1/jax/vae/vae.py` 中，定义了三种 VAE 配置：

```python
VAE_CONFIGS = {
    "flux": {
        "pretrained_vae_name_or_path": "black-forest-labs/FLUX.1-dev",
        "vae_channels": 16,
        "vae_compression_factor": 8,
    },
    "flux2": {
        "pretrained_vae_name_or_path": "black-forest-labs/FLUX.2-dev",
        "vae_channels": 32,
        "vae_compression_factor": 8,
    },
    "qwenimage": {
        "pretrained_vae_name_or_path": "Qwen/Qwen-Image",
        "vae_channels": 16,
        "vae_compression_factor": 8,
    },
}
```

#### 1.2 最终模型使用的 VAE 类型

在最终的 i1-3B 训练配置 `i1/jax/configs/i1_training/1024_resolution.py` 中：

```python
config.vae_type = "flux2"
```

这明确指定使用 FLUX.2 的 VAE。

#### 1.3 VAE 加载方式

在 `load_vae()` 函数中，FLUX.2 VAE 通过 `diffusers` 库加载：

```python
vae_pt = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.2-dev", subfolder="vae", ...
)
```

#### 1.4 VAE 结构详情

FLUX.2 VAE 是一个标准的 `AutoencoderKL`，其 Flax 实现在 `i1/jax/vae/flax_flux.py` 中：
- **Encoder**：`FlaxFluxEncoder` — Conv2D 输入层 → 多个下采样块（ResNet Block + Downsample）→ 中间块（ResNet + Attention）→ GroupNorm + Conv 输出
- **Decoder**：`FlaxFluxDecoder` — Conv2D 输入层 → 中间块 → 多个上采样块（ResNet Block + Upsample）→ GroupNorm + Conv 输出
- **Latent 通道数**：32 通道
- **空间压缩比**：8x（即 1024×1024 图像 → 128×128 的 latent）
- **使用 Diagonal Gaussian Distribution**：encoder 输出 mean 和 logvar，采样得到 latent

#### 1.5 Latent 归一化

FLUX.2 VAE 使用特殊的 latent 归一化方式（`scale_latents` / `reverse_scale_latents`）：
- 将 32 通道 latent 在空间维度上做 2×2 packing，变成 128 通道
- 使用预计算的 per-channel mean 和 variance 进行标准化
- 再 unpack 回 32 通道

这与 FLUX.1 VAE（使用简单的 shift/scale 常数）不同，FLUX.2 采用了更精细的 per-channel 归一化。

---

## 2. DIT 模型分析

### 问题：这个模型是否使用了 flow_matching 的 DIT 模型？若是，使用的是单流DIT还是双流MMDIT？

### 结论：**i1 使用了基于 Rectified Flow（Flow Matching）的 DIT 模型，最终采用的是双流 MMDiT（Multimodal DiT）架构。**

### 代码分析依据

#### 2.1 Flow Matching（Rectified Flow）确认

在 `i1/jax/diffusion/rectified_flow.py` 中：

```python
def prepare_rectified_flow_inputs(latents, noise_key, time_key, cfg):
    x1 = latents          # 目标（clean latent）
    x0 = jax.random.normal(noise_key, latents.shape, dtype=latents.dtype)  # 噪声
    t = sample_times(time_key, latents.shape[0], cfg, dtype=latents.dtype)
    t_expanded = _broadcast_t(t, latents.shape)
    xt = (1.0 - t_expanded) * x0 + t_expanded * x1   # 线性插值
    ut = x1 - x0                                       # velocity target
    return xt, ut, t
```

这是标准的 Rectified Flow 公式：
- `x_t = (1-t) * x_0 + t * x_1`（线性插值路径）
- `u_t = x_1 - x_0`（速度场目标）
- 训练目标：预测 velocity `u_t`

配置中也确认：
```python
config.transport = mlc.ConfigDict(dict(
    prediction="velocity",
    use_lognorm=True,
    ...
))
```

#### 2.2 双流 MMDiT 确认

在配置中：
```python
config.backbone = "dual_stream"
config.model_size = 'DiT-XL_2016'
```

代码中同时实现了三种 backbone：
1. **`single_stream`**（`single_stream_backbone.py`）：文本和图像 token 拼接后共享 QKV 权重
2. **`dual_stream`**（`dual_stream_backbone.py`）：文本和图像 token 使用**独立的 QKV 和 MLP 权重**，但在 attention 计算时做**联合注意力**
3. **`cross_attn`**（`cross_attn_backbone.py`）：图像做 self-attention，文本通过 cross-attention 注入

最终 i1-3B 模型使用 `dual_stream` — 这就是双流 MMDiT 架构。

#### 2.3 MMDiT 双流机制

在 `MMDiTAttention` 中（`dual_stream_backbone.py`）：

```python
# 图像和文本使用独立的 QKV 投影
self.qkv_image = nn.Dense(3 * self.hidden_size, ...)
self.qkv_text = nn.Dense(3 * self.hidden_size, ...)

# 但在 attention 时拼接在一起做联合注意力
q = jnp.concatenate([q_image, q_text], axis=2)
k = jnp.concatenate([k_image, k_text], axis=2)
v = jnp.concatenate([v_image, v_text], axis=2)

# 输出使用独立的线性投影
self.out_image = nn.Dense(self.hidden_size, ...)
self.out_text = nn.Dense(self.hidden_size, ...)
```

---

## 3. 子网络结构

### 问题：这个模型的具体网络结构是怎样的？一共有哪些子网络结构？

### 结论：i1 模型包含以下子网络结构：

| 子网络 | 名称 | 描述 | 关键参数 |
|--------|------|------|----------|
| ① | **VAE Encoder** | FLUX.2 AutoencoderKL 的 Encoder 部分 | 输入: 3通道RGB → 输出: 32通道 latent，空间压缩8x |
| ② | **VAE Decoder** | FLUX.2 AutoencoderKL 的 Decoder 部分 | 输入: 32通道 latent → 输出: 3通道RGB |
| ③ | **Text Encoder** | T5Gemma-2B-2B（预训练冻结） | 输出维度: 2304，最大 token 数: 256 |
| ④ | **Text Encoder Adapter** | 2层 Transformer 连接器 | 将 2304 维映射到 2016 维（DIT hidden_size） |
| ⑤ | **Timestep Embedder** | 正弦位置编码 + MLP | 256维频率编码 → 2016维 |
| ⑥ | **DIT Backbone（双流 MMDiT）** | 29层双流 MMDiT Block，带 U-Net Long Skip | hidden_size=2016, 28 heads, SwiGLU FFN |
| ⑦ | **Final Layer** | 归一化 + 线性投影 | 2016维 → patch_size² × 32 通道 |

### 3.1 各子网络详细结构

#### ① VAE Encoder（训练时使用，推理时不需要）
```
输入图像 [B, 3, H, W]
  → Conv2D(3→128) 
  → DownBlock(128→128) + Downsample
  → DownBlock(128→256) + Downsample  
  → DownBlock(256→512) + Downsample
  → DownBlock(512→512)
  → MidBlock(512, ResNet+Attention+ResNet)
  → GroupNorm + SiLU + Conv2D(512→64)  [double_z=True, 64=32×2]
  → DiagonalGaussianDistribution → 采样
输出 latent [B, 32, H/8, W/8]
```

#### ② VAE Decoder（推理时使用）
```
输入 latent [B, 32, H/8, W/8]
  → post_quant_conv(32→32)
  → Conv2D(32→512)
  → MidBlock(512)
  → UpBlock(512→512) + Upsample
  → UpBlock(512→256) + Upsample
  → UpBlock(256→128) + Upsample
  → UpBlock(128→128)
  → GroupNorm + SiLU + Conv2D(128→3)
输出图像 [B, 3, H, W]
```

#### ③ Text Encoder（T5Gemma-2B-2B，预训练冻结）
```
输入文本 tokens [B, 256]
  → T5Gemma Encoder
输出 hidden states [B, 256, 2304]
```

#### ④ Text Encoder Adapter（2层 Transformer 连接器）
```
输入 text embeddings [B, 256, 2304]
  → Linear(2304→2016)    [connector_in]
  → Block 1:
    → RMSNorm → Attention → 残差连接
    → RMSNorm → SwiGLU FFN → 残差连接
  → Block 2:
    → RMSNorm → Attention → 残差连接
    → RMSNorm → SwiGLU FFN → 残差连接
输出 adapted text tokens [B, 256, 2016]
```

同时包含一个 `learnable_null_caption`（可学习的空文本嵌入）用于 CFG。

#### ⑤ Timestep Embedder
```
输入 timestep t [B]
  → 正弦频率编码(256维)
  → Linear(256→2016) + SiLU + Linear(2016→2016)
输出 timestep embedding [B, 2016]
```

#### ⑥ DIT Backbone（双流 MMDiT + U-Net Long Skip）

**整体结构：**
```
image_tokens + text_tokens
  → 14个 In Blocks（双流 MMDiT Block，无 skip）
  → 1个 Mid Block（双流 MMDiT Block）
  → 14个 Out Blocks（双流 MMDiT Block，带 skip 连接）
```

**单个 DualStreamDiTBlock 结构（无 AdaLN 模式）：**
```
image_tokens, text_tokens:
  → RMSNorm(image), RMSNorm(text)        [norm1]
  → MMDiTAttention:
      image_qkv = qkv_image(normed_image)
      text_qkv = qkv_text(normed_text)
      QKNorm → Multimodal RoPE
      联合 attention（拼接 q,k,v 计算）
      image_attn = proj_image(attn[:image])
      text_attn = proj_text(attn[image:])
  → Sandwich Norm(image_attn), Sandwich Norm(text_attn)   [norm3]
  → 残差连接
  → RMSNorm(image), RMSNorm(text)        [norm2]
  → SwiGLU_image(normed_image), SwiGLU_text(normed_text)
  → Sandwich Norm(mlp_out)                [norm4]
  → 残差连接
输出 updated image_tokens, text_tokens
```

**Out Block 额外包含 skip 连接：**
```
image_tokens = skip_linear_image(cat[image_tokens, skip_image])
text_tokens = skip_linear_text(cat[text_tokens, skip_text])
→ 然后执行正常的 DualStreamDiTBlock 计算
```

#### ⑦ Final Layer（无 AdaLN 版本）
```
输入 image_tokens [B, N, 2016]
  → RMSNorm(2016)
  → Linear(2016 → patch_size² × 32) = Linear(2016 → 128)
输出 [B, N, 128]
  → Reshape/Unpatchify → [B, 32, H/8, W/8]
```

---

## 4. 模型网络结构图

### 问题：通过箭头来简单示意由各个子网络结构组成的模型网络结构图。

### 结论：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        i1 模型整体结构                               │
│                                                                     │
│   ┌──────────┐         ┌──────────────┐                             │
│   │ 输入图像  │ ──────→ │ VAE Encoder  │ ──→ latent z               │
│   │(训练时)   │         │ (FLUX.2 AE)  │    [B,32,H/8,W/8]         │
│   └──────────┘         └──────────────┘                             │
│                                                                     │
│   ┌──────────┐                                                      │
│   │ 文本提示  │ ──────→ ┌──────────────┐     ┌────────────────────┐ │
│   │ (Prompt) │         │ Text Encoder │ ──→ │ Text Encoder       │ │
│   └──────────┘         │ (T5Gemma)    │     │ Adapter (2层Trans) │ │
│                        └──────────────┘     └────────┬───────────┘ │
│                                                      │             │
│                                          text_tokens [B,256,2016]  │
│                                                      │             │
│   ┌──────────┐     ┌────────────────┐                │             │
│   │ 时间步 t  │ ──→ │ Timestep       │ ──→ cond      │             │
│   └──────────┘     │ Embedder       │   [B,2016]     │             │
│                    └────────────────┘     │           │             │
│                                          │           │             │
│   ┌──────────┐     ┌────────────────┐    │           │             │
│   │ 噪声 x_t │ ──→ │ Patch Embed +  │ ──→ image_tokens            │
│   │[B,32,    │     │ Pos Embed +    │   [B,N,2016]   │             │
│   │ H/8,W/8] │     │ Multimodal RoPE│                │             │
│   └──────────┘     └────────────────┘    │           │             │
│                                          ↓           ↓             │
│                    ┌─────────────────────────────────────────────┐  │
│                    │        DIT Backbone (双流 MMDiT)             │  │
│                    │                                             │  │
│                    │  ┌─────────────────┐                        │  │
│                    │  │ In Block 0      │ ──→ skip_0             │  │
│                    │  │ (MMDiT Block)   │                        │  │
│                    │  ├─────────────────┤                        │  │
│                    │  │ In Block 1      │ ──→ skip_1             │  │
│                    │  ├─────────────────┤                        │  │
│                    │  │ ...             │                        │  │
│                    │  ├─────────────────┤                        │  │
│                    │  │ In Block 13     │ ──→ skip_13            │  │
│                    │  ├─────────────────┤                        │  │
│                    │  │ Mid Block       │                        │  │
│                    │  ├─────────────────┤                        │  │
│                    │  │ Out Block 0     │ ←── skip_13            │  │
│                    │  ├─────────────────┤                        │  │
│                    │  │ ...             │                        │  │
│                    │  ├─────────────────┤                        │  │
│                    │  │ Out Block 13    │ ←── skip_0             │  │
│                    │  └─────────────────┘                        │  │
│                    └──────────────┬──────────────────────────────┘  │
│                                  │                                  │
│                                  ↓                                  │
│                    ┌─────────────────────┐                          │
│                    │ Final Layer         │                          │
│                    │ (RMSNorm + Linear)  │                          │
│                    └──────────┬──────────┘                          │
│                               │                                     │
│                               ↓                                     │
│                    predicted velocity v_θ                            │
│                    [B, 32, H/8, W/8]                                │
│                               │                                     │
│                    (推理时: ODE 积分得到 clean latent)                │
│                               │                                     │
│                               ↓                                     │
│                    ┌──────────────────┐                              │
│                    │ VAE Decoder      │                              │
│                    │ (FLUX.2 AE)      │                              │
│                    └──────────┬───────┘                              │
│                               │                                     │
│                               ↓                                     │
│                    输出图像 [B, 3, H, W]                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. 模型能力分析

### 问题：这个模型能否实现文生图？能否实现图像+文本提示进行图像编辑？

### 结论：
- **文生图**：✅ **能够实现**。这是 i1 模型的核心功能。
- **图像+文本提示进行图像编辑**：❌ **不能实现**。

### 代码分析依据

#### 5.1 文生图能力确认

从 `torch_inference/generate.py` 的 `denoise_latents()` 函数可以看出：

```python
def denoise_latents(model, text, mask, args, device):
    shape = (text.shape[0], 32, model.input_size, model.input_size)
    latents = torch.randn(shape, ...)  # 从纯噪声开始
    ...
    for idx in tqdm(range(args.num_steps), desc="Denoising"):
        ...
        latents = latents + (times[idx + 1] - times[idx]) * velocity
    return latents
```

输入仅需文本提示，从随机噪声开始，通过 ODE 积分生成图像。

#### 5.2 图像编辑能力不存在的依据

1. **DIT 前向传播接口**：`i1DiT.forward(x, t, caption, mask)` 的输入只有噪声 latent `x`、时间步 `t`、文本 `caption` 和注意力掩码 `mask`，**没有任何接受参考图像的输入接口**。

2. **推理代码中没有图像编辑入口**：`torch_inference/generate.py` 只提供了文生图的命令行参数（`--prompt`、`--prompt-set` 等），没有任何图像输入参数。

3. **训练配置中无图像编辑数据**：配置文件中的训练数据只包含 image-caption 对，没有图像编辑的 triplet 数据（source image, instruction, target image）。

4. **与 FLUX.2 的对比**：FLUX.2 模型通过 `forward_kv_extract()` 和 `forward_kv_cached()` 支持参考图像编辑（Reference Image Token 机制），但 i1 的 DIT 代码中完全没有此类机制。

---

## 6. 文生图流程图

### 问题：画出文生图时的完整数据流程。

### 结论：

```
═══════════════════════════════════════════════════════════════════════
                        文生图完整流程
═══════════════════════════════════════════════════════════════════════

输入数据:
  ├── 文本提示 (prompt): str
  └── 随机种子 (seed): int

═══════════════ 第一阶段：文本编码 ═══════════════

  文本提示 (str)
       │
       ↓
  ┌─────────────────┐
  │ Tokenizer       │  (T5Gemma tokenizer)
  │ max_length=256  │
  └────────┬────────┘
           │
    input_ids [B, 256]
    attention_mask [B, 256]
           │
           ↓
  ┌─────────────────┐
  │ Text Encoder    │  (T5Gemma-2B-2B, 冻结权重)
  │ (子网络③)       │
  └────────┬────────┘
           │
    hidden_states [B, 256, 2304]
    attention_mask [B, 256]
           │
           ↓
  ┌─────────────────────────┐
  │ Text Encoder Adapter    │  (2层 Transformer 连接器)
  │ (子网络④)               │
  │ Linear(2304→2016)       │
  │ → TransformerBlock × 2  │
  └────────┬────────────────┘
           │
    text_tokens [B, 256, 2016]
    text_mask [B, 256]

═══════════════ 第二阶段：CFG 准备 ═══════════════

  text_tokens [B, 256, 2016]
       │
       ↓
  ┌───────────────────────────────┐
  │ 拼接条件/无条件文本嵌入       │
  │ cond_text = text_tokens       │
  │ uncond_text = learnable_null  │ ← 可学习的空文本嵌入参数
  │ cfg_text = cat[cond, uncond]  │ → [2B, 256, 2016]
  │ cfg_mask = cat[mask, mask]    │ → [2B, 256]
  └───────────────┬───────────────┘
                  │
  ┌───────────────────────────────┐
  │ 预计算 RoPE 位置编码          │
  │ MultimodalRopeEmbedder:       │
  │   axes = (time, row, col)     │
  │   text_freqs, image_freqs    │
  │ → 缓存为 forward_cache       │
  └───────────────┬───────────────┘
                  │
         forward_cache (text_tokens, text_mask, 
                        image_freqs, text_freqs)

═══════════════ 第三阶段：去噪循环 ═══════════════

  latents = randn([B, 32, input_size, input_size])
  times = time_grid(num_steps=250, shift=0.3)  
  guidance = cfg_scale (默认12.0)

  FOR idx in range(num_steps):
      │
      │  t = times[idx]
      │  latent_input = cat[latents, latents]  → [2B, ...]
      │  t_input = cat[t, t]                   → [2B]
      │
      │         ↓
      │  ┌─────────────────────────────────────────────┐
      │  │ i1DiT Forward (子网络⑤⑥⑦)                  │
      │  │                                             │
      │  │  latent_input [2B, 32, H/8, W/8]            │
      │  │       │                                     │
      │  │       ↓                                     │
      │  │  PatchEmbed + sinusoidal pos_embed          │
      │  │  → image_tokens [2B, N, 2016]               │
      │  │                                             │
      │  │  使用 forward_cache 中预计算的:              │
      │  │  text_tokens, text_mask, image/text_freqs   │
      │  │       │                                     │
      │  │       ↓                                     │
      │  │  ┌─────────────────────────┐                │
      │  │  │ DIT Backbone            │                │
      │  │  │ 14 In Blocks            │                │
      │  │  │ → 1 Mid Block           │                │
      │  │  │ → 14 Out Blocks (skip)  │                │
      │  │  │ (每个Block:             │                │
      │  │  │  MMDiT Joint Attention  │                │
      │  │  │  + Sandwich Norm        │                │
      │  │  │  + SwiGLU FFN)          │                │
      │  │  └──────────┬──────────────┘                │
      │  │             │                               │
      │  │  只取 image_tokens                          │
      │  │       │                                     │
      │  │       ↓                                     │
      │  │  ┌─────────────────┐                        │
      │  │  │ Final Layer     │                        │
      │  │  │ RMSNorm+Linear  │                        │
      │  │  └────────┬────────┘                        │
      │  │           │                                 │
      │  │  Unpatchify → velocity [2B, 32, H/8, W/8]  │
      │  └─────────────────────────────────────────────┘
      │
      │  velocity_cond, velocity_uncond = split(velocity)
      │  
      │  # CFG 引导
      │  velocity = cond + (cfg_scale - 1) * (cond - uncond)
      │  
      │  # CFG Rescale（可选, 默认 phi=1.0）
      │  factor = std(cond) / std(velocity)
      │  velocity = velocity * (1-phi + phi*factor)
      │
      │  # Euler 步进
      │  latents = latents + (times[idx+1] - times[idx]) * velocity
      │
  END FOR

═══════════════ 第四阶段：VAE 解码 ═══════════════

  clean_latents [B, 32, H/8, W/8]
       │
       ↓
  ┌──────────────────────────┐
  │ Reverse Scale Latents    │
  │ (FLUX.2 latent 反归一化)  │
  │ 2×2 unpack → 128ch       │
  │ × sqrt(var) + mean        │
  │ 2×2 pack → 32ch          │
  └────────────┬─────────────┘
               │
       ↓
  ┌──────────────────────────┐
  │ VAE Decoder (子网络②)    │
  │ post_quant_conv          │
  │ → Conv + MidBlock        │
  │ → UpBlocks × 4           │
  │ → GroupNorm + SiLU + Conv │
  └────────────┬─────────────┘
               │
  decoded_image [B, 3, H, W]
       │
       ↓
  (image / 2 + 0.5).clamp(0, 1)  → 归一化到 [0,1]
       │
       ↓
  × 255 → uint8 → PIL Image → 保存为 PNG

═══════════════ 输出 ═══════════════

  输出: 生成的图像 (PNG 文件)
  最终输出来自: VAE Decoder (子网络②)
```

---

## 7. 图像编辑流程图

### 问题：若模型能实现图像+文本提示进行图像编辑，画出流程图。

### 结论：**i1 模型不支持图像+文本提示进行图像编辑，因此无法画出此流程图。**

详细原因参见 [第5节分析](#52-图像编辑能力不存在的依据)。

i1 是一个纯粹的 **Text-to-Image（文生图）** 模型。其设计目标是探索文本到图像扩散模型的设计空间，提供一个简单且完全开源的高性能文生图方案。代码中没有任何机制支持：
- 图像条件输入（image conditioning）
- 参考图像 token（reference image tokens）
- 图像编辑指令（editing instructions）
- Inpainting / Outpainting
- Image-to-Image 翻译

---

## 8. 相比 FLUX.2 模型的创新点

### 问题：相比于FLUX.2模型，本模型具有哪些创新点或改进点？

### 结论：以下是通过代码级对比分析得出的所有创新/改进点：

---

### 8.1 去除 AdaLN（Adaptive Layer Norm），采用简单 Pre-Norm

**FLUX.2**：使用 AdaLN（Adaptive Layer Normalization），由 timestep embedding 生成 shift、scale、gate 参数来调制每一层的归一化：
```python
# FLUX.2 的 Modulation
class Modulation(nn.Module):
    def forward(self, vec):
        out = self.lin(F.silu(vec))  # 6倍hidden_size
        return shift, scale, gate  # AdaLN 调制参数
```

**i1**：完全去除 AdaLN（`use_adaln=False`），使用简单的 Pre-Norm（先归一化再处理），不对每层进行自适应调制：
```python
# i1 配置
use_adaln=False
# 实际代码中：直接 norm → attention → 残差
image_attn, text_attn = attn(norm1_image(image_tokens), norm1_text(text_tokens), ...)
image_tokens = image_tokens + image_attn
```

**意义**：大幅简化模型结构，减少参数量和计算量，论文探索表明在足够的训练下 AdaLN 并非必要。

---

### 8.2 引入 Sandwich Norm（三明治归一化）

**FLUX.2**：不使用 Sandwich Norm。

**i1**：在 attention 输出和 MLP 输出之后额外添加归一化层（`use_sandwich_norm=True`）：
```python
# i1 的 DualStreamDiTBlock
image_attn = self.norm3(image_attn)   # Sandwich Norm on attention output
text_attn = self.norm3(text_attn)
image_tokens = image_tokens + image_attn

image_mlp = self.norm4(image_mlp)     # Sandwich Norm on MLP output
text_mlp = self.norm4(text_mlp)
image_tokens = image_tokens + image_mlp
```

**意义**：增强训练稳定性，防止残差连接中特征幅度的累积增长。

---

### 8.3 使用 SwiGLU FFN 替代 SiLU 门控 MLP

**FLUX.2**：使用 `SiLUActivation`（SiLU 门控机制）：
```python
class SiLUActivation(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2  # SiLU 门控
```

**i1**：使用 SwiGLU FFN（`use_swiglu=True`）：
```python
class SwiGLUFFN(nn.Module):
    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)  # SwiGLU
```

**区别**：SwiGLU 在门控之后还有一个输出投影层 `w3`，而 FLUX.2 的 SiLU 门控直接输出。SwiGLU 使用 2/3 的 hidden_features 维度来补偿额外的参数开销。

---

### 8.4 使用 RMSNorm 替代 LayerNorm

**FLUX.2**：attention block 内部使用 `nn.LayerNorm(elementwise_affine=False)`，QK norm 使用 `RMSNorm`。

**i1**：全面使用 `RMSNorm`（`use_rmsnorm=True`），包括 block 内的所有归一化层：
```python
class RMSNorm(nn.Module):
    def forward(self, x):
        x_float = x.float()
        x_float = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * self.scale.float()).to(dtype)
```

**意义**：RMSNorm 比 LayerNorm 计算效率更高（不需要计算 mean），且在大规模模型中效果相当。

---

### 8.5 引入 U-Net 风格的 Long Skip 连接

**FLUX.2**：不使用 skip 连接。29 层双流 + 48 层单流顺序排列，没有 U-Net 结构。

**i1**：使用 U-Net 风格的 Long Skip 连接（`use_long_skip=True`）：
```python
# 29层 = 14 In + 1 Mid + 14 Out (with skip)
num_in_blocks = depth // 2  # 14
self.in_blocks = [DualStreamDiTBlock(...) for _ in range(14)]
self.mid_block = DualStreamDiTBlock(...)
self.out_blocks = [DualStreamDiTBlock(..., use_skip=True) for _ in range(14)]

# Forward:
skips = []
for blk in self.in_blocks:
    image_tokens, text_tokens = blk(...)
    skips.append((image_tokens, text_tokens))
image_tokens, text_tokens = self.mid_block(...)
for blk in self.out_blocks:
    image_tokens, text_tokens = blk(..., skip=skips.pop())
```

Skip 连接通过线性层融合：
```python
image_tokens = skip_linear_image(cat[image_tokens, skip_image])  # 2*hidden → hidden
text_tokens = skip_linear_text(cat[text_tokens, skip_text])
```

**意义**：借鉴 U-Net 的成功经验，通过跳跃连接帮助保留低层特征，改善生成质量。

---

### 8.6 使用 Transformer 连接器作为 Text Encoder Adapter

**FLUX.2**：使用简单的线性投影将文本编码器输出映射到模型隐空间：
```python
self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)
```

**i1**：使用 2 层 Transformer Block 作为 Text Encoder Adapter（`text_encoder_adapter_type="transformer"`, `text_encoder_adapter_num_blocks=2`）：
```python
class TextEncoderAdapterTransformer(nn.Module):
    # Linear(in_channels → hidden_size) 
    # + 2 × (RMSNorm → Attention → RMSNorm → SwiGLU FFN)
```

**意义**：更强的文本到视觉特征的适配能力，能够更好地对齐文本语义与视觉特征空间。

---

### 8.7 不同的 Text Encoder 选择

**FLUX.2**：使用多个 text encoder 的拼接方案（context_in_dim=15360，暗示拼接了多个大型 text encoder 的输出）。

**i1**：使用单一的 **T5Gemma-2B-2B** 作为 text encoder（hidden_dim=2304）：
```python
config.text_encoder_type = "T5Gemma"
# 推理代码中:
text_encoder = T5GemmaModel.from_pretrained("google/t5gemma-2b-2b-ul2-it").encoder
```

同时代码也支持多种其他 text encoder 选择（T5、CLIP、Qwen3、FG-CLIP2、T5Gemma2 等），提供了灵活的实验框架。

**意义**：简化 text encoder 部分，降低推理成本，同时通过系统性实验证明单一高质量 text encoder 也能取得好效果。

---

### 8.8 不同的位置编码策略：Sinusoidal + Multimodal RoPE

**FLUX.2**：仅使用 RoPE（4轴：每轴 32 维，theta=2000）：
```python
axes_dim: list[int] = [32, 32, 32, 32]  # 4轴
theta: int = 2000
```

**i1**：使用 **Sinusoidal 位置编码 + Multimodal RoPE** 双重方案（`position_embedding="sinusoidal_and_rope"`，3轴：time/row/col，theta=10000）：
```python
# Sinusoidal 可学习位置编码（初始化为2D sinusoidal）
self.pos_embed = nn.Parameter(...)  # 加到 image tokens 上

# Multimodal RoPE（3轴：time, row, col）
axes_dims = _default_rope_axes_dims(head_dim)  # 自动分配
# 时间轴 = text_num_tokens + 1，空间轴 = hw × hw
# 支持分辨率内插（256基础 → 512/1024 缩放）
```

**差异**：
- FLUX.2 用 4 轴 RoPE（包含一个额外维度），i1 用 3 轴（time, row, col）
- i1 额外使用 sinusoidal 绝对位置编码
- i1 的 theta=10000 vs FLUX.2 的 theta=2000
- i1 支持基于 256 基础分辨率的位置编码内插

---

### 8.9 不同的 CFG（Classifier-Free Guidance）策略

**FLUX.2**：使用 **Guidance Embedding**（条件引导标量编码为 embedding 直接加到 timestep embedding 上）：
```python
self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
# Forward:
vec = self.time_in(timestep_emb) + self.guidance_in(guidance_emb)
```

**i1**：使用 **可学习的 Null Caption + 标准 CFG**：
```python
# 训练时：以 drop_text_prob 概率替换 caption 为 learnable_null_caption
self.learnable_null_caption = nn.Parameter(torch.empty(1, token_len, in_channels))

# 推理时：条件和无条件预测做引导
velocity = cond + (cfg_scale - 1) * (cond - uncond)
```

同时 i1 还支持 **CFG Rescale**：
```python
# 对引导后的 velocity 做标准差校正
std_c = std(cond)
std_g = std(velocity)
factor = std_c / (std_g + 1e-8)
velocity = velocity * (1 - phi + phi * factor)
```

**意义**：标准 CFG 更加通用，CFG Rescale 可以缓解过度引导导致的色彩饱和问题。

---

### 8.10 不同的时间步采样策略

**FLUX.2**：使用基于图像序列长度的自适应 SNR 偏移（`generalized_time_snr_shift`）：
```python
def generalized_time_snr_shift(t, mu, sigma):
    return exp(mu) / (exp(mu) + (1/t - 1)**sigma)

# mu 根据图像像素数动态计算
mu = compute_empirical_mu(image_seq_len, num_steps)
```

**i1**：使用 **LogNormal 采样 + 线性 Shift**：
```python
# 训练时：LogNormal 采样
normal_samples = lognorm_mu + lognorm_sigma * random.normal(...)
t = sigmoid(normal_samples)

# 推理时：均匀网格 + 线性 shift (shift=0.3)
times = linspace(0, 1, num_steps + 1)
times = (shift * times) / (1 + (shift - 1) * times)
```

**意义**：LogNormal 采样使训练更多关注中间噪声水平，线性 shift 在推理时调整时间步分布以改善生成质量。

---

### 8.11 纯双流架构 vs 混合双流+单流架构

**FLUX.2**：使用**「先双流后单流」的混合架构**：
```python
# FLUX.2: 8层双流 + 48层单流
depth: int = 8                    # 双流层数
depth_single_blocks: int = 48     # 单流层数
```
先用 8 层 DoubleStreamBlock 分别处理图像和文本，然后拼接后用 48 层 SingleStreamBlock 处理。

**i1**：使用**纯双流架构**：
```python
# i1: 29层全双流
depth: int = 29   # 全部是双流 DualStreamDiTBlock
```
所有 29 层都是 DualStreamDiTBlock，图像和文本始终保持独立的 QKV 和 MLP。

**意义**：更一致的架构设计，通过 U-Net skip 连接补偿深度不足。

---

### 8.12 大幅缩小的模型规模

**FLUX.2**：
- hidden_size = 6144，48 heads
- 8 双流层 + 48 单流层 = 56 层总计
- 参数量约 **12B+**

**i1**：
- hidden_size = 2016，28 heads
- 29 双流层（14 in + 1 mid + 14 out）
- 参数量 **3B**（或 1B 变体）

**意义**：在显著更小的模型规模下，通过精心的架构和训练设计，在多个基准上取得与领先模型竞争力的性能。

---

### 8.13 完全开源的训练方案

**FLUX.2**：仅开源推理代码和模型权重，不开源训练代码和数据。

**i1**：**完全开源**所有内容：
- ✅ JAX 和 PyTorch 训练代码
- ✅ JAX 和 PyTorch 推理代码
- ✅ 训练数据（i1-captions 数据集）
- ✅ 数据处理流水线（下载、重标注、TFRecord 创建）
- ✅ 所有中间检查点（256、512、1024 分辨率）
- ✅ 完整的训练配方和超参数

**意义**：为社区提供完全可复现的文生图模型训练方案。

---

### 8.14 不使用 Guidance Embedding

**FLUX.2**：使用额外的 Guidance Embedding 层，将 CFG 引导强度编码为模型条件输入：
```python
self.use_guidance_embed = True
self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
```

**i1**：不使用 Guidance Embedding。CFG 仅在推理时通过外部条件/无条件预测的线性组合实现，模型本身不感知引导强度。

---

### 8.15 条件信号注入方式不同

**FLUX.2**：timestep embedding 通过 Modulation（AdaLN）注入到每一层，生成 shift/scale/gate 来调制：
```python
# 全局共享的 Modulation
self.double_stream_modulation_img = Modulation(hidden_size, double=True)
self.double_stream_modulation_txt = Modulation(hidden_size, double=True)
self.single_stream_modulation = Modulation(hidden_size, double=False)
```

**i1**：由于去除了 AdaLN，timestep 信息仅通过与 pooled text embedding 相加形成 `cond`，但在 `use_adaln=False` 模式下**不用于调制 DIT 层**。DIT 层完全不依赖 timestep 条件（仅 Final Layer 在 AdaLN 模式下使用 cond）。在 i1 的最终配置中，DIT 内部的 transformer blocks 是**无条件的**（不受 timestep 影响），仅通过文本 token 的联合注意力引入条件。

---

### 8.16 不支持图像编辑（Reference Image Token）

**FLUX.2**：通过 `forward_kv_extract()` 和 `forward_kv_cached()` 支持参考图像 token，实现图像编辑功能。包含 causal attention mask 使参考 token 仅自注意。

**i1**：不支持任何形式的图像条件输入或编辑功能，是纯粹的文生图模型。

---

### 创新点总结表

| # | 创新/改进点 | FLUX.2 | i1 |
|---|-----------|--------|-----|
| 1 | AdaLN | ✅ 使用 | ❌ 去除 |
| 2 | Sandwich Norm | ❌ | ✅ 使用 |
| 3 | FFN 类型 | SiLU 门控 | SwiGLU |
| 4 | 归一化类型 | LayerNorm (无参数) | RMSNorm (有 scale 参数) |
| 5 | Skip 连接 | ❌ | ✅ U-Net Long Skip |
| 6 | Text Adapter | 简单线性投影 | 2层 Transformer |
| 7 | Text Encoder | 多 encoder 拼接 (15360维) | 单一 T5Gemma (2304维) |
| 8 | 位置编码 | 4轴 RoPE | Sinusoidal + 3轴 RoPE |
| 9 | CFG 方式 | Guidance Embedding | Learnable Null Caption + CFG Rescale |
| 10 | 时间步采样 | SNR Shift | LogNormal + Linear Shift |
| 11 | 架构组合 | 8层双流 + 48层单流 | 29层纯双流 |
| 12 | 模型规模 | ~12B+ 参数 | 3B 参数 |
| 13 | 开源程度 | 仅推理代码+权重 | 完全开源（训练代码+数据+配方） |
| 14 | Guidance Embedding | ✅ 使用 | ❌ 不使用 |
| 15 | 条件注入 | 逐层 Modulation | 仅通过联合注意力 |
| 16 | 图像编辑 | ✅ 支持 (Ref Token + KV Cache) | ❌ 不支持 |
