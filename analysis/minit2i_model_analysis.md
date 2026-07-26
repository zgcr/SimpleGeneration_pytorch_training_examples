# MiniT2I 模型全面分析报告

> 分析对象：`minit2i-torch` 代码仓库
> 分析时间：2026年6月

> **代码实现根目录**：`/root/code/minit2i-torch/`

---

## 目录

1. [是否使用了VAE？](#1-是否使用了vae)
2. [是否使用了flow_matching的DIT模型？](#2-是否使用了flow_matching的dit模型)
3. [具体网络结构与子网络划分](#3-具体网络结构与子网络划分)
4. [模型网络结构图](#4-模型网络结构图)
5. [模型能力：文生图与图像编辑](#5-模型能力文生图与图像编辑)
6. [文生图流程图](#6-文生图流程图)
7. [图像编辑流程图](#7-图像编辑流程图)
8. [与FLUX2模型的创新点对比](#8-与flux2模型的创新点对比)

---

## 1. 是否使用了VAE？

### 结论：MiniT2I **没有使用VAE**，它是一个纯像素空间（pixel-space）的直接生成模型。

### 代码依据

**（1）模型输入输出通道为RGB 3通道**

在 `mini_t2i/model.py` 中，`MMJiTB32Text2` 的默认参数：

```python
class MMJiTB32Text2(nn.Module):
    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 32,
        in_channels: int = 3,  # ← 直接使用RGB 3通道
        ...
    ):
```

在 `diffusers/mmdit.py` 中的 `MMJiTConfig`：

```python
@dataclass
class MMJiTConfig:
    image_size: int = 512
    patch_size: int = 16
    in_channels: int = 3   # ← 直接使用RGB 3通道
```

**（2）训练时直接操作原始像素图像**

在 `mini_t2i/train.py` 的 `prepare_images` 函数中：

```python
def prepare_images(pixel_values: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = pixel_values.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = images.to(dtype=torch.float32).mul_(1.0 / 127.5).add_(-1.0)  # ← 仅做归一化到[-1,1]
    return images
```

没有任何VAE编码步骤，图像直接从uint8转为float32后归一化。

**（3）采样时直接输出像素图像**

在 `mini_t2i/diffusion.py` 的 `euler_sample` 函数中：

```python
x = torch.randn(b, 3, image_size, image_size, ...) * noise_scale  # ← 3通道噪声
...
return x.clamp(-1, 1)  # ← 直接输出像素空间图像
```

没有任何VAE解码步骤。

**（4）整个代码库中完全没有VAE相关代码**

搜索整个 `minit2i-torch` 目录，没有任何 `autoencoder`、`vae`、`encoder`/`decoder`（指VAE的encoder/decoder）、`latent` 等关键词。

**（5）README明确说明**

> "MiniT2I is a simple direct-RGB text-to-image generator that trains a **pixel-space** MM-JiT denoiser with flow matching...The recipe is intentionally plain: **avoiding image tokenizers**, cascaded generation, RL stages, and any auxiliary losses."

### 与FLUX1/FLUX2 VAE的对比

- **FLUX1** 的 VAE 使用标准的 convolutional autoencoder，`z_channels=16`，patch size `[2,2]`，即8倍下采样，latent通道 `16*2*2=64`。
- **FLUX2** 的 VAE 在代码中可见 `z_channels=32`，patch size `[2,2]`，latent通道 `32*2*2=128`（`in_channels=128`），并使用 `BatchNorm` 进行归一化/反归一化替代传统的 KL 正则化。
- **MiniT2I** 完全不使用VAE，直接在 `3` 通道像素空间操作。

---

## 2. 是否使用了flow_matching的DIT模型？

### 结论：MiniT2I **使用了基于 flow matching 的双流 MMDIT（Multi-Modal Joint Transformer）模型**。

### 2.1 Flow Matching 证据

**（1）训练损失函数（`mini_t2i/diffusion.py`）**

```python
def training_loss(...):
    t = sample_lognorm(b, t_lognorm_mu, t_lognorm_sigma, device)  # 对数正态分布采样时间步
    noise = torch.randn_like(images) * noise_scale
    x_t = images * t[:, None, None, None] + noise * (1.0 - t[:, None, None, None])  # ← flow matching 插值
    pred_x0 = model(x_t, t, text_embeddings, attention_mask)
    target = (images - x_t) / (1.0 - t[:, None, None, None]).clamp_min(0.05)  # ← velocity target
    v_pred = (pred_x0 - x_t) / (1.0 - t[:, None, None, None]).clamp_min(0.05)  # ← velocity prediction
    per_sample = (v_pred - target).pow(2).mean(dim=(1, 2, 3))
    loss = per_sample.mean()
```

这是标准的 **Rectified Flow / Flow Matching** 框架：
- 使用线性插值 `x_t = x_1 * t + x_0 * (1-t)` 构建轨迹（其中 `x_1=images`, `x_0=noise`）
- 预测 `x_0`（clean image），然后计算velocity误差
- 时间步从对数正态分布采样

**（2）Euler采样器（`mini_t2i/diffusion.py`）**

```python
def euler_sample(...):
    x = torch.randn(...) * noise_scale
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)  # ← 从0到1的时间步
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        ...
        v = (pred_x0 - x) / (1.0 - t).clamp_min(0.05)
        x = x + v * (t1 - t0)  # ← Euler ODE积分
    return x.clamp(-1, 1)
```

标准的 Euler ODE 求解器，沿 flow matching 轨迹积分。

### 2.2 双流MMDIT结构证据

**（1）`DoubleStreamBlock` 类（`mini_t2i/model.py`）**

```python
class DoubleStreamBlock(nn.Module):
    def __init__(self, ...):
        # 图像流和文本流各有独立的norm、QKV投影、MLP
        self.img_norm1 = RMSNorm(hidden_size)
        self.img_norm2 = RMSNorm(hidden_size)
        self.txt_norm1 = RMSNorm(hidden_size)
        self.txt_norm2 = RMSNorm(hidden_size)
        self.img_qkv = nn.Linear(hidden_size, inner * 3)
        self.txt_qkv = nn.Linear(hidden_size, inner * 3)
        # 共享的QK归一化
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        # 各自的投影和MLP
        self.img_proj = nn.Linear(inner, hidden_size)
        self.txt_proj = nn.Linear(inner, hidden_size)
        self.img_mlp = SwiGLU(hidden_size, ...)
        self.txt_mlp = SwiGLU(hidden_size, ...)

    def forward(self, img, txt):
        # 各自独立计算QKV
        qi, ki, vi = self.img_qkv(self.img_norm1(img))...
        qt, kt, vt = self.txt_qkv(self.txt_norm1(txt))...
        # 拼接后做联合注意力
        q = torch.cat([qt, qi], dim=1)
        k = torch.cat([kt, ki], dim=1)
        v = torch.cat([vt, vi], dim=1)
        # 联合注意力计算
        out = attention_forward(q, k, v, ...)
        # 拆分回各自的流
        txt = txt + self.txt_proj(out[:, :lt])
        img = img + self.img_proj(out[:, lt:])
        # 各自独立的MLP
        img = img + self.img_mlp(self.img_norm2(img))
        txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return img, txt
```

这是典型的 **双流（Double-Stream）MMDIT** 结构：
- 图像和文本各有独立的 QKV 投影和 MLP（双流）
- 注意力计算时将两个流的 Q/K/V 拼接在一起做联合注意力（Multi-Modal Joint Attention）
- 注意力输出后拆分回各自的流

**（2）没有单流block**

与 FLUX2 不同，MiniT2I **仅使用双流block**，没有 `SingleStreamBlock`。模型只有 `DoubleStreamBlock` 堆叠。

---

## 3. 具体网络结构与子网络划分

MiniT2I 的整体架构由 **2个子网络** 组成：

### 子网络1：T5 文本编码器（FLAN-T5-Large）

| 属性 | 值 |
|------|-----|
| 模型 | `google/flan-t5-large` |
| 参数量 | ~341M |
| 隐藏维度 | 1024 |
| 训练状态 | 冻结（`freeze_t5=True`） |
| 输出 | `last_hidden_state`，shape `(B, L, 1024)` |
| 最大序列长度 | 256 tokens |

代码位置：`mini_t2i/train.py` 中的 `build_text_encoder()`

```python
def build_text_encoder(cfg, device):
    enc = T5EncoderModel.from_pretrained(cfg.t5_name, ...)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc
```

### 子网络2：MM-JiT 去噪器（核心DIT网络）

这是模型的核心，由以下子模块组成：

| 子模块 | 类名 | 作用 |
|--------|------|------|
| 图像Patch嵌入 | `BottleneckPatchEmbed` | 将像素图像切成patch并投影到隐藏空间 |
| 文本嵌入投影 | `nn.Linear` (txt_embed) | 将T5编码的1024维映射到hidden_size |
| 掩码token | `nn.Parameter` (mask_token) | 用于CFG中无条件生成时替换文本 |
| 时间步嵌入 | `TimestepEmbedder` | 将标量时间步t转为向量 |
| 池化文本嵌入 | `nn.Linear` (pooled_embed) | 将T5输出的均值池化映射到hidden_size |
| 正弦余弦位置编码 | `sincos_2d` | 2D位置编码（不可学习buffer） |
| 文本预处理块 | `PlainTextBlock` × 2 | 文本单独的自注意力+MLP块 |
| 双流注意力块 | `DoubleStreamBlock` × 17 | 图像-文本联合注意力+各自MLP |
| 最终归一化 | `RMSNorm` (final_norm) | 输出前的归一化 |
| 最终线性层 | `nn.Linear` (final) | 将hidden_size映射到patch_size²×3 |

**B/32模型参数配置**（训练代码中的默认配置）：

| 参数 | 值 |
|------|-----|
| image_size | 512 |
| patch_size | 32 |
| in_channels | 3 |
| hidden_size | 768 |
| t5_hidden_size | 1024 |
| depth_double | 17 |
| text_preamble_depth | 2 |
| num_heads | 12 |
| head_dim | 64 |
| mlp_ratio | 2.6667 |
| pca_channels | 128 |

**B/16和L/16模型参数配置**（diffusers推理代码中的配置）：

| 参数 | B/16 | L/16 |
|------|------|------|
| image_size | 512 | 512 |
| patch_size | 16 | 16 |
| hidden_size | 768 | 1248 |
| num_heads | 12 | 24 |
| head_dim | 64 | 52 |
| 总参数量 | ~258M | ~912M |

### BottleneckPatchEmbed 详细结构

```python
class BottleneckPatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, hidden_size, pca_channels):
        # 第1步：大卷积切patch + PCA降维（3→128）
        self.proj1 = nn.Conv2d(in_channels, pca_channels, kernel_size=patch_size, stride=patch_size, bias=False)
        # 第2步：1x1卷积升维（128→768）
        self.proj2 = nn.Conv2d(pca_channels, hidden_size, kernel_size=1, bias=True)
```

这是一个两步式的瓶颈结构，先降维再升维，而非一步直接映射。

### PlainTextBlock 详细结构

```python
class PlainTextBlock(nn.Module):
    # 标准的Transformer自注意力块，仅处理文本
    # Pre-norm (RMSNorm) + Self-Attention + Residual
    # Pre-norm (RMSNorm) + SwiGLU MLP + Residual
    # 使用1D RoPE位置编码
    # 使用QK归一化（RMSNorm）
```

### DoubleStreamBlock 详细结构

```python
class DoubleStreamBlock(nn.Module):
    # 双流设计：
    # - 图像流：img_norm1 → img_qkv → 联合注意力 → img_proj → img_norm2 → img_mlp
    # - 文本流：txt_norm1 → txt_qkv → 联合注意力 → txt_proj → txt_norm2 → txt_mlp
    # 注意力计算：
    #   Q = cat([Q_text, Q_img]), K = cat([K_text, K_img]), V = cat([V_text, V_img])
    #   文本用1D RoPE，图像用2D RoPE
    # QK归一化：共享的q_norm和k_norm（RMSNorm）
```

---

## 4. 模型网络结构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MiniT2I 整体架构                             │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐   │
│  │ 子网络1:       │    │ 子网络2: MM-JiT 去噪器                   │   │
│  │ T5文本编码器   │    │                                          │   │
│  │ (FLAN-T5-L)  │    │  ┌──────────────────────────────────┐    │   │
│  │              │    │  │ BottleneckPatchEmbed              │    │   │
│  │ 文本prompt    │    │  │ (图像→patch tokens)               │    │   │
│  │    ↓         │    │  └──────────────────────────────────┘    │   │
│  │ Tokenizer    │    │                                          │   │
│  │    ↓         │    │  ┌──────────────────────────────────┐    │   │
│  │ T5 Encoder   │───→│  │ txt_embed + pooled_embed         │    │   │
│  │    ↓         │    │  │ (文本投影)                         │    │   │
│  │ text_emb     │    │  └──────────────────────────────────┘    │   │
│  │ (B,L,1024)   │    │                                          │   │
│  └──────────────┘    │  ┌──────────────────────────────────┐    │   │
│                      │  │ TimestepEmbedder                  │    │   │
│  噪声图像 x_t ──────→│  │ (时间步嵌入)                       │    │   │
│                      │  └──────────────────────────────────┘    │   │
│  时间步 t ──────────→│                                          │   │
│                      │  ┌──────────────────────────────────┐    │   │
│                      │  │ PlainTextBlock × 2                │    │   │
│                      │  │ (文本预处理自注意力)                │    │   │
│                      │  └──────────────────────────────────┘    │   │
│                      │                                          │   │
│                      │  ┌──────────────────────────────────┐    │   │
│                      │  │ DoubleStreamBlock × 17            │    │   │
│                      │  │ (图像流 + 文本流 联合注意力)        │    │   │
│                      │  │  ┌─────────┐  ┌─────────┐        │    │   │
│                      │  │  │ 图像流   │  │ 文本流   │        │    │   │
│                      │  │  │ img_qkv │  │ txt_qkv │        │    │   │
│                      │  │  │ img_mlp │  │ txt_mlp │        │    │   │
│                      │  │  └─────────┘  └─────────┘        │    │   │
│                      │  └──────────────────────────────────┘    │   │
│                      │                                          │   │
│                      │  ┌──────────────────────────────────┐    │   │
│                      │  │ FinalLayer                        │    │   │
│                      │  │ (RMSNorm → Linear → unpatchify)   │    │   │
│                      │  └──────────────────────────────────┘    │   │
│                      │        ↓                                 │   │
│                      │   输出：pred_x0 (B, 3, H, W)             │   │
│                      └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 简化箭头示意图

```
文本 prompt ──→ [T5 Tokenizer] ──→ [T5 Encoder] ──→ text_embeddings (B, L, 1024)
                                                          │
                                                          ├──→ [txt_embed] ──→ txt (B, L, 768)
                                                          │
                                                          └──→ mean pool ──→ [pooled_embed] ──→ vec
                                                                                                 │
时间步 t ──────────────────────→ [TimestepEmbedder] ──→ t_emb ──→ vec = t_emb + pooled    (del vec)
                                                                                                 
噪声图像 x_t ──→ [BottleneckPatchEmbed] ──→ img_tokens + pos_embed ──→ img
                                                                          │
                          txt ──→ [PlainTextBlock ×2] ──→ txt'            │
                                                           │              │
                                                           ↓              ↓
                                                    [DoubleStreamBlock ×17]
                                                    (联合注意力: Q/K/V拼接)
                                                           │              │
                                                           │              ↓
                                                           │    [RMSNorm → Linear]
                                                           │              │
                                                           │              ↓
                                                           │    [unpatchify]
                                                           │              │
                                                           │              ↓
                                                           │    pred_x0 (B, 3, 512, 512)
```

**注意**：代码中 `vec`（时间步嵌入 + 池化文本嵌入）在计算后被 `del vec` 删除，实际并未被传入任何transformer block中用于调制。这意味着 MiniT2I 的 DIT block 不使用 AdaLN 调制机制。

---

## 5. 模型能力：文生图与图像编辑

### 5.1 文生图能力

**结论：✅ MiniT2I 能够实现文生图。**

代码依据：

1. **训练流程**（`mini_t2i/train.py`）：输入文本+图像对，训练去噪模型。
2. **采样流程**（`mini_t2i/diffusion.py` 的 `euler_sample`）：仅需要文本输入即可生成图像。
3. **Diffusers Pipeline**（`diffusers/pipeline.py`）：`MiniT2ITextToImagePipeline.__call__()` 接收 `prompt` 参数生成图像。
4. **LoRA微调**（`lora/train_lora.py`）和 **评估pipeline**（`mini_t2i/eval_pipeline.py`）均证实文生图功能。
5. **README** 明确说明："MiniT2I is a simple direct-RGB **text-to-image** generator"。

### 5.2 图像编辑能力

**结论：❌ MiniT2I 不能实现图像+文本提示进行图像编辑。**

代码依据：

1. **模型forward签名**：`forward(self, img, t, context, attn_mask)` —— `img` 是噪声图像 `x_t`，不是参考/编辑源图像。
2. **采样流程**：`euler_sample` 从纯随机噪声开始，没有任何接收参考图像的参数。
3. **数据加载**：训练数据只包含 `pixel_values`（目标图像）、`input_ids`（文本token）和 `attention_mask`，没有参考图像输入。
4. **整个代码库**没有任何：
   - 参考图像编码逻辑
   - 图像编辑相关的采样方法（如SDEdit、图像条件注入等）
   - Image-to-image 或 inpainting 接口
5. **与FLUX2对比**：FLUX2 的 `forward_kv_extract` 和 `forward_kv_cached` 方法支持参考图像token的注入（通过 `x_seq_concat` 参数），而 MiniT2I 完全没有此类设计。

---

## 6. 文生图流程图

### 训练阶段流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        训练阶段流程                                  │
│                                                                     │
│  输入数据:                                                           │
│  ├── pixel_values: 原始RGB图像 (B, 3, 512, 512) [uint8]             │
│  ├── input_ids: 文本token ids (B, 256)                              │
│  └── attention_mask: 注意力掩码 (B, 256)                             │
│                                                                     │
│  Step 1: 图像预处理                                                  │
│  pixel_values ──→ [归一化到[-1,1]] ──→ images (B, 3, 512, 512)      │
│                                                                     │
│  Step 2: 文本编码 (冻结的T5)                                         │
│  input_ids + attention_mask ──→ [T5 Encoder] ──→ text_emb (B,256,1024)│
│                                                                     │
│  Step 3: 采样时间步                                                  │
│  t = sigmoid(randn * σ + μ)  ──→ t ∈ (0, 1)                        │
│                                                                     │
│  Step 4: 构造噪声图像                                                │
│  noise = randn_like(images) * 2.0                                   │
│  x_t = images * t + noise * (1 - t)                                 │
│                                                                     │
│  Step 5: 模型前向传播                                                │
│  x_t, t, text_emb, attn_mask ──→ [MM-JiT去噪器] ──→ pred_x0        │
│                                                                     │
│  Step 6: 计算velocity损失                                           │
│  target_v = (images - x_t) / (1 - t)                                │
│  pred_v = (pred_x0 - x_t) / (1 - t)                                │
│  loss = MSE(pred_v, target_v)                                       │
│                                                                     │
│  Step 7: 反向传播 + 优化器更新 + EMA更新                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 推理/采样阶段流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        推理/采样阶段流程                              │
│                                                                     │
│  输入数据:                                                           │
│  └── text prompt: 文本描述字符串                                      │
│                                                                     │
│  Step 1: 文本编码                                                    │
│  text ──→ [T5 Tokenizer] ──→ input_ids (1, 256)                     │
│                              attention_mask (1, 256)                 │
│  input_ids ──→ [T5 Encoder] ──→ text_emb (1, 256, 1024)             │
│                                                                     │
│  Step 2: 初始化纯噪声                                                │
│  x_0 = randn(1, 3, 512, 512) * noise_scale                         │
│                                                                     │
│  Step 3: Euler ODE积分 (100步)                                       │
│  for i = 0, 1, ..., 99:                                             │
│      t_i = i / 100, t_{i+1} = (i+1) / 100                          │
│      │                                                               │
│      │  有CFG (cfg_scale ≠ 1.0):                                     │
│      │  pred_cond = MM-JiT(x, t, text_emb, mask)                    │
│      │  pred_uncond = MM-JiT(x, t, text_emb, null_mask)             │
│      │  pred_x0 = pred_uncond + cfg_scale * (pred_cond - pred_uncond)│
│      │                                                               │
│      │  无CFG (cfg_scale = 1.0):                                     │
│      │  pred_x0 = MM-JiT(x, t, text_emb, mask)                     │
│      │                                                               │
│      v = (pred_x0 - x) / (1 - t)                                    │
│      x = x + v * (t_{i+1} - t_i)                                    │
│                                                                     │
│  Step 4: 后处理                                                      │
│  output = x.clamp(-1, 1)                                            │
│  image = (output * 127.5 + 128.0).to(uint8)                         │
│                                                                     │
│  最终输出:                                                           │
│  └── RGB图像 (H, W, 3) [uint8, 0-255]                               │
│      来自 MM-JiT 去噪器的最终预测                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### MM-JiT 去噪器内部前向传播详细流程

```
输入: img (B,3,512,512), t (B,), context (B,256,1024), attn_mask (B,256)

  Step 1: 文本条件掩码处理
  ├── mask = attn_mask.bool()[:, :, None]
  └── context = where(mask, context, mask_token)    # CFG时用mask_token替换

  Step 2: 图像patch嵌入
  ├── img ──→ [Conv2d(3→128, k=32, s=32)] ──→ [Conv2d(128→768, k=1)] ──→ flatten
  └── img_tokens = patch_embed + pos_embed           # shape: (B, 16*16, 768) = (B, 256, 768)

  Step 3: 文本嵌入
  ├── txt = txt_embed(context)                       # Linear(1024→768): (B, 256, 768)
  ├── pooled = context.mean(dim=1)                   # (B, 1024)
  └── vec = t_embed(t) + pooled_embed(pooled)        # (B, 768)  ← 注意: vec随后被del

  Step 4: 文本预处理
  └── for block in txt_blocks (×2):
          txt = PlainTextBlock(txt)                   # 自注意力 + SwiGLU MLP

  Step 5: 双流联合注意力
  └── for block in blocks (×17):
          img_tokens, txt = DoubleStreamBlock(img_tokens, txt)
          │
          ├── img/txt 各自计算 QKV
          ├── Q = cat([Q_txt, Q_img]),  K = cat([K_txt, K_img]),  V = cat([V_txt, V_img])
          ├── 文本Q/K: 1D RoPE,  图像Q/K: 2D RoPE
          ├── QK归一化 (RMSNorm)
          ├── 联合注意力计算
          ├── 拆分: txt_out = out[:, :L_txt],  img_out = out[:, L_txt:]
          ├── img = img + img_proj(img_out),  txt = txt + txt_proj(txt_out)
          └── img = img + img_mlp(norm(img)),  txt = txt + txt_mlp(norm(txt))

  Step 6: 最终输出
  ├── out = final_linear(final_norm(img_tokens))     # Linear(768→32*32*3=3072)
  └── return unpatchify(out)                          # (B, 3, 512, 512)
```

---

## 7. 图像编辑流程图

### 结论：不适用

如第5节所分析，MiniT2I **不支持图像+文本提示进行图像编辑**，因此无法绘制图像编辑的流程图。

MiniT2I 是一个纯粹的文生图（Text-to-Image）模型，其设计目标是作为一个极简基线（minimalist baseline），刻意避免了复杂功能。整个代码库中没有任何与图像编辑、image-to-image、inpainting、参考图像条件注入等相关的代码实现。

如需图像编辑功能，需要参考 FLUX2 等支持参考图像条件的模型架构。

---

## 8. 与FLUX2模型的创新点对比

以下逐一列举 MiniT2I 相比 FLUX2 的所有创新点和改进点（以及显著差异点）：

### 8.1 无VAE的像素空间直接生成（最核心创新）

| 特性 | MiniT2I | FLUX2 |
|------|---------|-------|
| 生成空间 | **像素空间（pixel-space）** | 潜在空间（latent-space） |
| VAE | **无** | 有（`z_channels=32`, `in_channels=128`） |
| 输入/输出通道 | 3 (RGB) | 128 (latent) |

MiniT2I 完全去除了 VAE 编码器/解码器，直接在 RGB 像素空间进行去噪。这是最大的创新点，极大简化了训练和推理流程：
- 训练时无需预先编码图像到latent空间
- 推理时无需VAE解码
- 消除了VAE引入的信息损失和重建误差
- 减少了整体模型组件和参数量

### 8.2 仅使用双流block（去除单流block）

| 特性 | MiniT2I | FLUX2 |
|------|---------|-------|
| 双流block数量 | **17** | 8 |
| 单流block数量 | **0** | 48 |
| 总block数量 | 17+2(text) = 19 | 8+48 = 56 |

MiniT2I 移除了 FLUX2 中大量的单流（Single-Stream）block，仅保留双流block。这意味着图像和文本在整个处理过程中始终保持独立的表示流，通过联合注意力交互但不合并。

### 8.3 轻量级文本编码器

| 特性 | MiniT2I | FLUX2 |
|------|---------|-------|
| 文本编码器 | **FLAN-T5-Large (~341M)** | Mistral-Small-3.2-24B (~24B) 或 Qwen3 |
| 编码器类型 | **纯文本编码器** | 多模态大语言模型 |
| 文本嵌入维度 | **1024** | 15360 (3层×5120) |
| 最大token数 | 256 | 512 |
| 提示增强 | **无** | 有（使用LLM进行prompt upsampling） |

MiniT2I 使用小型的纯文本编码器，而非 FLUX2 那样使用数十亿参数的多模态大语言模型。这极大降低了推理成本和显存需求。

### 8.4 瓶颈式Patch Embedding（两步卷积PCA降维）

| 特性 | MiniT2I | FLUX2 |
|------|---------|-------|
| Patch嵌入 | **两步卷积（先降维再升维）** | 单步线性投影 |
| 中间瓶颈维度 | **128 (pca_channels)** | 无瓶颈 |
| 结构 | `Conv2d(3→128, k=32) → Conv2d(128→768, k=1)` | `Linear(128→hidden_size)` |

MiniT2I 的 `BottleneckPatchEmbed` 使用两步卷积：
1. 第一步大卷积核切patch同时降维到128维（类似PCA降维）
2. 第二步1×1卷积升维到hidden_size

这是一个参数高效的设计，利用瓶颈结构减少初始投影的参数量。

### 8.5 无AdaLN调制机制

| 特性 | MiniT2I | FLUX2 |
|------|---------|-------|
| 条件注入方式 | **无显式调制** | AdaLN（Adaptive LayerNorm）调制 |
| timestep使用 | 计算后被 `del` 删除 | 通过Modulation层生成shift/scale/gate |
| Norm类型 | **RMSNorm** | LayerNorm (elementwise_affine=False) + AdaLN |

这是一个非常值得注意的设计选择。在 `mini_t2i/model.py` 的 `MMJiTB32Text2.forward()` 中：

```python
vec = self.t_embed(t).to(dtype=img.dtype) + self.pooled_embed(pooled)
del vec  # ← 计算了但立即删除！
```

MiniT2I 计算了时间步嵌入和池化文本嵌入的组合向量 `vec`，但随后立即删除，**不将其传入任何transformer block进行调制**。这意味着：
- 时间步信息仅作为模型输入 `t` 的直接参数传入模型（在训练时作为插值系数），但不通过AdaLN调制影响网络内部计算。
- 文本条件完全通过联合注意力机制注入，而非AdaLN调制。

FLUX2 则广泛使用 Modulation 层，从 `vec` 生成 shift/scale/gate 用于每个block的 AdaLN 调制。

### 8.6 RMSNorm + SwiGLU（vs LayerNorm + SiLU gated）

| 特性 | MiniT2I | FLUX2 |
|------|---------|-------|
| 归一化 | **RMSNorm**（可选 elementwise_affine） | LayerNorm（elementwise_affine=False） |
| MLP激活 | **SwiGLU** (SiLU gate × linear) | SiLU gated（类似SwiGLU但实现略不同） |
| MLP结构 | `w1, w3 → SiLU(w1(x)) * w3(x) → w2` | `linear1 → [qkv, mlp_split] → SiLU gate → linear2` |

MiniT2I 统一使用 RMSNorm 而非 LayerNorm，计算更高效（无需计算均值）。

### 8.7 2D RoPE位置编码方式不同

| 特性 | MiniT2I | FLUX2 |
|------|---------|-------|
| 图像位置编码 | **2D RoPE (flat)** + **sincos_2d固定位置** | 4D位置ID + RoPE (`axes_dim=[32,32,32,32]`) |
| 文本位置编码 | **1D RoPE** | 4D位置ID + RoPE |
| 位置编码基础 | **固定sincos_2d (buffer)** + 旋转编码 | 纯旋转编码 |
| RoPE实现 | 直接在forward中计算 | 通过 `EmbedND` 模块 |

MiniT2I 使用两层位置编码：
1. 固定的 `sincos_2d` 作为加性位置编码（加到patch embedding上）
2. 注意力中使用 2D RoPE 旋转位置编码

FLUX2 则使用统一的4维位置ID系统（t, h, w, l），通过 `EmbedND` 计算RoPE。

### 8.8 QK归一化使用RMSNorm（vs 专用QKNorm模块）

| 特性 | MiniT2I | FLUX2 |
|------|---------|-------|
| QK归一化 | **RMSNorm (shared q_norm, k_norm)** | QKNorm (内含 query_norm, key_norm 的 RMSNorm) |
| 共享方式 | 图像流和文本流共享同一对 q_norm/k_norm | 每个attention有独立的QKNorm |

### 8.9 无Guidance Embedding

| 特性 | MiniT2I | FLUX2 |
|------|---------|-------|
| Guidance嵌入 | **无** | 有（`guidance_in = MLPEmbedder`） |
| CFG实现 | 通过两次前向传播+线性插值 | 支持guidance embedding + 双次前向CFG |

MiniT2I 的CFG通过经典的双次前向传播实现（条件+无条件），而 FLUX2 额外支持通过 guidance embedding 实现单次前向的隐式指导。

### 8.10 无参考图像/编辑能力

| 特性 | MiniT2I | FLUX2 |
|------|---------|-------|
| 参考图像输入 | **不支持** | 支持（`x_seq_concat` + KV缓存） |
| 图像编辑 | **不支持** | 支持（通过参考图像token注入） |
| KV缓存加速 | **无** | 有（`forward_kv_extract` + `forward_kv_cached`） |
| 因果注意力 | **无** | 有（参考token仅自注意力） |

FLUX2 设计了完整的参考图像处理流程：
- 通过VAE编码参考图像
- 参考token通过 `x_seq_concat` 拼接到图像序列
- 使用因果注意力（参考token仅自注意力，不attend到目标图像）
- 首步提取KV缓存，后续步骤复用

MiniT2I 完全没有此类设计。

### 8.11 极简设计理念

| 特性 | MiniT2I | FLUX2 |
|------|---------|-------|
| 设计目标 | **极简基线 (minimalist baseline)** | 商业级全功能模型 |
| 辅助损失 | **无** | 可能有 |
| 级联生成 | **无** | 无 |
| RL/RLHF | **无** | 可能有 |
| 总参数量 | **258M~912M** (+ 341M T5) | 数十亿 (+ 24B LLM) |
| 训练数据 | **公开数据** (CC12M + 120K mix) | 大规模私有数据 |

MiniT2I 的核心价值在于其极简设计，作为研究基线（baseline）：
- 仅使用公开数据训练
- 不使用任何辅助技巧
- 代码简洁可读
- 资源需求低

### 8.12 噪声缩放因子

| 特性 | MiniT2I | FLUX2 |
|------|---------|-------|
| 初始噪声缩放 | **`noise_scale=2.0`** | 默认1.0 |
| 训练噪声 | `noise * noise_scale` | 标准正态 |

MiniT2I 使用 `noise_scale=2.0` 对训练和采样中的初始噪声进行缩放，这是一个特殊的超参数选择。

### 8.13 时间步采样策略

| 特性 | MiniT2I | FLUX2 |
|------|---------|-------|
| 训练时间步采样 | **对数正态分布 (μ=-0.8, σ=0.8)** | 经验性μ-shifted schedule |
| 推理时间步 | **均匀线性 [0, 1]** | SNR-shifted schedule (基于图像分辨率) |

MiniT2I 训练时使用简单的对数正态分布采样时间步，推理时使用均匀线性时间步。FLUX2 使用基于图像序列长度的自适应schedule。

### 8.14 文本预处理块

| 特性 | MiniT2I | FLUX2 |
|------|---------|-------|
| 文本预处理 | **PlainTextBlock ×2** | 无独立文本预处理 |
| 预处理内容 | 自注意力 + SwiGLU MLP | — |

MiniT2I 在文本进入双流block之前，先通过2个独立的 `PlainTextBlock` 进行预处理，让文本表示在参与图文联合注意力之前先进行自注意力交互。FLUX2 没有此类独立的文本预处理步骤。

### 创新点总结表

| 编号 | 创新/改进点 | 影响 |
|------|------------|------|
| 1 | 无VAE像素空间直接生成 | 简化流程、消除VAE误差 |
| 2 | 仅双流block（去除单流block） | 减少参数、简化架构 |
| 3 | 轻量T5文本编码器替代大型LLM | 大幅降低资源需求 |
| 4 | 瓶颈式Patch Embedding（PCA降维） | 参数高效 |
| 5 | 无AdaLN调制（vec被del） | 简化条件注入机制 |
| 6 | 统一RMSNorm替代LayerNorm | 计算更高效 |
| 7 | 独特的2D RoPE + sincos_2d双层位置编码 | 位置信息增强 |
| 8 | 共享QK归一化 | 减少参数 |
| 9 | 无Guidance Embedding | 简化CFG实现 |
| 10 | 无参考图像/编辑功能 | 专注文生图核心能力 |
| 11 | 极简设计理念（无辅助损失/RL/级联） | 可复现、可研究 |
| 12 | 噪声缩放因子noise_scale=2.0 | 特殊超参数选择 |
| 13 | 对数正态分布时间步采样 | 训练稳定性 |
| 14 | 文本预处理PlainTextBlock | 增强文本表示质量 |

---

*本分析基于 `minit2i-torch` 代码仓库的完整源码阅读和 `flux2` 代码的对比分析得出。*
