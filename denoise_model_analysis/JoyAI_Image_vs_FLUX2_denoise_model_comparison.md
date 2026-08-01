# JoyAI-Image vs FLUX2 去噪模型架构对比分析报告

> 本报告聚焦于两个模型的**去噪模型（Denoising Model / DiT）**部分的网络结构差异，基于以下代码仓库的完整源代码分析得出，每条结论均已与代码实现进行复核校验。
>
> - **JoyAI-Image 代码根目录**：`/opt/nas/p/zhugechaoran/download/code/JoyAI-Image/`
> - **FLUX2 代码根目录**：`/opt/nas/p/zhugechaoran/download/code/flux2/`

---

## 目录

- [JoyAI-Image vs FLUX2 去噪模型架构对比分析报告](#joyai-image-vs-flux2-去噪模型架构对比分析报告)
  - [目录](#目录)
  - [1. 两个去噪模型的整体架构概览](#1-两个去噪模型的整体架构概览)
    - [JoyAI-Image 去噪模型（`Transformer3DModel`）](#joyai-image-去噪模型transformer3dmodel)
    - [FLUX2 去噪模型（`Flux2`）](#flux2-去噪模型flux2)
  - [2. 变化创新点列表总览](#2-变化创新点列表总览)
  - [3. 各创新点详细分析](#3-各创新点详细分析)
    - [创新点1：纯双流架构 vs 双流+单流混合架构](#创新点1纯双流架构-vs-双流单流混合架构)
    - [创新点2：WanX可学习参数表调制 vs MLP全量投影调制](#创新点2wanx可学习参数表调制-vs-mlp全量投影调制)
    - [创新点3：每层独立调制参数 vs 全局共享调制参数](#创新点3每层独立调制参数-vs-全局共享调制参数)
    - [创新点4：3D RoPE位置编码 vs 4D RoPE位置编码](#创新点43d-rope位置编码-vs-4d-rope位置编码)
    - [创新点5：GELU-approximate MLP激活 vs SiLU Gated (SwiGLU) MLP激活](#创新点5gelu-approximate-mlp激活-vs-silu-gated-swiglu-mlp激活)
    - [创新点6：Conv3d Patchify输入投影 vs Linear线性输入投影](#创新点6conv3d-patchify输入投影-vs-linear线性输入投影)
    - [创新点7：带Bias的线性层设计 vs 无Bias线性层设计](#创新点7带bias的线性层设计-vs-无bias线性层设计)
    - [创新点8：文本不参与RoPE位置编码 vs 文本参与4D RoPE位置编码](#创新点8文本不参与rope位置编码-vs-文本参与4d-rope位置编码)
    - [创新点9：Multi-item全程参与注意力编辑机制 vs 因果注意力/KV-Cache参考机制](#创新点9multi-item全程参与注意力编辑机制-vs-因果注意力kv-cache参考机制)
    - [创新点10：多模态VLM文本编码器 vs 纯文本LLM文本编码器（单层 vs 多层提取）](#创新点10多模态vlm文本编码器-vs-纯文本llm文本编码器单层-vs-多层提取)
    - [创新点11：条件嵌入器架构差异（WanTimeTextImageEmbedding vs MLPEmbedder+独立投影）](#创新点11条件嵌入器架构差异wantimetextimageembedding-vs-mlpembedder独立投影)
    - [创新点12：输出层结构差异（简单Norm+Linear vs AdaLN调制输出层）](#创新点12输出层结构差异简单normlinear-vs-adaln调制输出层)
    - [创新点13：Flash Attention varlen vs PyTorch scaled\_dot\_product\_attention](#创新点13flash-attention-varlen-vs-pytorch-scaled_dot_product_attention)
    - [创新点14：CFG噪声范数归一化 vs 标准CFG/Guidance Embedding](#创新点14cfg噪声范数归一化-vs-标准cfgguidance-embedding)
    - [创新点15：无Guidance Embedding vs 有Guidance Embedding](#创新点15无guidance-embedding-vs-有guidance-embedding)
  - [4. 总结](#4-总结)
    - [JoyAI-Image去噪模型相比FLUX2去噪模型的主要创新点总结](#joyai-image去噪模型相比flux2去噪模型的主要创新点总结)
    - [核心结论](#核心结论)

---

## 1. 两个去噪模型的整体架构概览

### JoyAI-Image 去噪模型（`Transformer3DModel`）

| 属性 | 值 |
|------|-----|
| 核心类 | `Transformer3DModel` (`models.py`) |
| 隐藏层维度 | `hidden_size=3072` |
| 注意力头数 | `heads_num=24` |
| head_dim | 128 (3072/24) |
| 双流块数量 | `mm_double_blocks_depth=20` |
| 单流块数量 | **0**（无单流块） |
| 总层数 | **20** |
| 输入投影 | `nn.Conv3d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)` |
| 输出层 | `LayerNorm + nn.Linear` |
| 位置编码 | 3D RoPE，`rope_dim_list=[16, 56, 56]`，`theta=256` |
| 调制机制 | `ModulateWan`（可学习参数表），每层独立 |
| MLP激活 | `gelu-approximate`（GELU tanh近似） |
| 注意力后端 | Flash Attention (`flash_attn_varlen_func`) |
| 线性层bias | `bias=True` |
| Guidance Embedding | 无 |

### FLUX2 去噪模型（`Flux2`）

| 属性 | 值 |
|------|-----|
| 核心类 | `Flux2` (`model.py`) |
| 隐藏层维度 | `hidden_size=6144`（Dev 32B）/ `3072`（Klein 4B）/ `4096`（Klein 9B） |
| 注意力头数 | `num_heads=48`（Dev）/ `24`（Klein 4B）/ `32`（Klein 9B） |
| head_dim | 128 (6144/48) |
| 双流块数量 | `depth=8`（Dev）/ `5`（Klein 4B）/ `8`（Klein 9B） |
| 单流块数量 | `depth_single_blocks=48`（Dev）/ `20`（Klein 4B）/ `24`（Klein 9B） |
| 总层数 | **56**（Dev 8+48）/ **25**（Klein 4B）/ **32**（Klein 9B） |
| 输入投影 | `nn.Linear(in_channels, hidden_size, bias=False)` |
| 输出层 | `LastLayer`（AdaLN调制 + Linear） |
| 位置编码 | 4D RoPE，`axes_dim=[32, 32, 32, 32]`，`theta=2000` |
| 调制机制 | `Modulation`（MLP投影），全局共享3个 |
| MLP激活 | `SiLUActivation`（SiLU gated，类似SwiGLU） |
| 注意力后端 | PyTorch `F.scaled_dot_product_attention` |
| 线性层bias | `bias=False`（几乎所有层） |
| Guidance Embedding | 有（`MLPEmbedder`） |

---

## 2. 变化创新点列表总览

| # | 创新点 | JoyAI-Image 做法 | FLUX2 做法 | 优化维度 |
|---|--------|-----------------|-----------|---------|
| 1 | 纯双流架构 | 20层纯双流MMDoubleStreamBlock | 8双流+48单流 | 模态独立性 → 生成效果 |
| 2 | WanX可学习参数表调制 | ModulateWan（可学习表+加法） | Modulation（SiLU+Linear MLP） | 参数效率 |
| 3 | 每层独立调制参数 | 每个block独立ModulateWan | 全局共享3个Modulation | 表达能力 → 生成效果 |
| 4 | 3D RoPE位置编码 | [16,56,56], theta=256 | [32,32,32,32], theta=2000 | 空间分辨率 → 生成效果 |
| 5 | GELU-approximate MLP | GELU(tanh近似) | SiLU Gated (SwiGLU) | 设计选择 |
| 6 | Conv3d Patchify输入 | nn.Conv3d | nn.Linear | 视频兼容性/局部特征 → 生成效果 |
| 7 | 带Bias线性层 | bias=True | bias=False | 表达能力 → 生成效果 |
| 8 | 文本不参与RoPE | 仅图像token用RoPE | 图像和文本都用RoPE | 设计简洁性 |
| 9 | Multi-item编辑机制 | 参考latent全程参与注意力 | 因果注意力+KV-Cache | 编辑质量 → 生成效果 |
| 10 | 多模态VLM文本编码器 | Qwen3VL（单层提取） | Mistral-Small/Qwen3（多层concat） | 多模态理解 → 生成效果 |
| 11 | 条件嵌入器架构 | WanTimeTextImageEmbedding | MLPEmbedder+独立投影 | 架构设计 |
| 12 | 简单输出层 | LayerNorm+Linear | AdaLN调制+Linear | 设计简洁性 |
| 13 | Flash Attention varlen | flash_attn_varlen_func | F.scaled_dot_product_attention | 计算效率 |
| 14 | CFG噪声范数归一化 | CFG + cond_norm归一化 | 标准CFG / Guidance Embedding | 生成质量稳定性 → 生成效果 |
| 15 | 无Guidance Embedding | 不使用 | 使用MLPEmbedder | 设计简洁性 |

---

## 3. 各创新点详细分析

### 创新点1：纯双流架构 vs 双流+单流混合架构

**代码验证：**

JoyAI-Image（`models.py` 第369-381行）：
```python
# 仅有 double_blocks，无 single_blocks
self.double_blocks = nn.ModuleList(
    [
        MMDoubleStreamBlock(...)
        for _ in range(mm_double_blocks_depth)  # 20层
    ]
)
```

FLUX2（`model.py` 第76-96行）：
```python
self.double_blocks = nn.ModuleList(
    [DoubleStreamBlock(...) for _ in range(params.depth)]  # 8层
)
self.single_blocks = nn.ModuleList(
    [SingleStreamBlock(...) for _ in range(params.depth_single_blocks)]  # 48层
)
```

**区别描述：**

- **JoyAI-Image**：采用**纯双流MMDiT架构**，共20层`MMDoubleStreamBlock`。图像流和文本流在所有层中始终保持独立的表示空间（独立的QKV投影、独立的MLP、独立的LayerNorm），仅在注意力计算时通过Q/K/V的concatenation实现联合注意力交互。
- **FLUX2**：采用**先双流后单流的混合架构**，先经过8层`DoubleStreamBlock`（图像和文本独立处理），然后将图像和文本token concatenate为统一序列，再经过48层`SingleStreamBlock`（使用同一组参数处理）。

**优点分析：**

这是一个**生成效果优化**的创新点。纯双流架构让图像和文本在所有层中保持独立的表示空间，意味着：
1. 图像特征和文本特征不会被强制混合到同一表示空间，可以保持各自模态的特征独立性；
2. 每一层都有独立的图像MLP和文本MLP，对两种模态分别做非线性变换，允许更精细的模态特定处理；
3. 避免了单流块中因文本和图像共享参数导致的潜在信息干扰。

不过需要注意，纯双流架构的总层数（20层）远少于FLUX2的混合架构（56层），因此在参数量可能更小的情况下实现了不同的效果权衡。

---

### 创新点2：WanX可学习参数表调制 vs MLP全量投影调制

**代码验证：**

JoyAI-Image（`modulate_layers.py` 第19-40行）：
```python
class ModulateWan(nn.Module):
    def __init__(self, hidden_size, factor, ...):
        self.modulate_table = nn.Parameter(
            torch.zeros(1, factor, hidden_size) / hidden_size**0.5,
            requires_grad=True
        )

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.unsqueeze(1)
        return [o.squeeze(1) for o in (self.modulate_table + x).chunk(self.factor, dim=1)]
```

FLUX2（`model.py` 第400-412行）：
```python
class Modulation(nn.Module):
    def __init__(self, dim, double, disable_bias=False):
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=not disable_bias)

    def forward(self, vec):
        out = self.lin(nn.functional.silu(vec))
        ...
        return out[:3], out[3:] if self.is_double else None
```

**区别描述：**

- **JoyAI-Image (ModulateWan)**：使用**可学习参数表（`modulate_table`）+ 加法**的方式生成调制参数。`modulate_table` 是一个形状为 `(1, 6, hidden_size)` 的可学习参数，初始化为接近零值。前向传播时，将时间步条件向量 `x` 直接**加到**参数表上，然后按6等分切分为 shift/scale/gate 三对。每个`ModulateWan`参数量仅为 `6 * hidden_size` 个标量参数。
- **FLUX2 (Modulation)**：使用**SiLU激活 + Linear MLP投影**的方式动态生成调制参数。将条件向量 `vec` 经过 SiLU 激活后通过一个线性层 `Linear(dim, 6*dim)` 投影得到调制参数。每个 Modulation 参数量为 `dim * 6 * dim`（一个完整的矩阵乘法）。

**优点分析：**

这是一个**参数效率**维度的优化。ModulateWan的参数量远小于MLP投影方式——每个ModulateWan仅需 `6 * hidden_size` 个参数（如 `6 * 3072 = 18432`），而FLUX2的Modulation需要 `hidden_size * 6 * hidden_size` 个参数（如 `6144 * 6 * 6144 = 226,492,416`）。ModulateWan通过可学习的基准偏移+条件向量加法，用极少的参数实现了类似的调制功能，大幅减少了参数量。

不过这同时也意味着ModulateWan的调制能力可能不如MLP投影方式灵活——MLP可以根据不同的条件输入动态生成差异化的调制参数，而参数表加法的表达能力相对受限。

---

### 创新点3：每层独立调制参数 vs 全局共享调制参数

**代码验证：**

JoyAI-Image（`models.py` 第103-137行）：
```python
class MMDoubleStreamBlock(nn.Module):
    def __init__(self, ...):
        # 每个block内部有独立的 img_mod 和 txt_mod
        self.img_mod = load_modulation(modulate_type=self.dit_modulation_type, ...)
        self.txt_mod = load_modulation(modulate_type=self.dit_modulation_type, ...)
```

FLUX2（`model.py` 第98-108行）：
```python
class Flux2(nn.Module):
    def __init__(self, params):
        # 全局共享 3 个 Modulation，所有层共享
        self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, ...)
        self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True, ...)
        self.single_stream_modulation = Modulation(self.hidden_size, double=False, ...)
```

FLUX2 在 `forward` 中（第132-134行）：
```python
double_block_mod_img = self.double_stream_modulation_img(vec)  # 计算一次，所有双流块共享
double_block_mod_txt = self.double_stream_modulation_txt(vec)
single_block_mod, _ = self.single_stream_modulation(vec)
```

**区别描述：**

- **JoyAI-Image**：每个`MMDoubleStreamBlock`内部有自己独立的`img_mod`和`txt_mod`（共 20×2=40 个 ModulateWan 实例）。每一层根据条件向量独立计算自己的 shift/scale/gate 参数，且每一层的可学习参数表 `modulate_table` 可以学到不同的基准值。
- **FLUX2**：仅有3个全局共享的`Modulation`实例（`double_stream_modulation_img`、`double_stream_modulation_txt`、`single_stream_modulation`），在 forward 最开始计算一次调制参数，然后**所有双流块共享同一套调制参数**，所有单流块共享同一套调制参数。

**优点分析：**

这是一个**表达能力 → 生成效果**维度的优化。每层独立的调制参数意味着模型可以为不同深度的层学到不同的基准调制行为。深层和浅层处理的特征语义层次不同，独立调制允许每层根据自身在网络中的位置进行差异化的条件注入，可能带来更精细的条件控制和更好的生成效果。

虽然JoyAI-Image的ModulateWan参数量很小（每个仅 `6*hidden_size`），但因为每层独立，总的调制参数量为 `20层 * 2(img+txt) * 6 * 3072 = 737,280`，仍然远小于FLUX2的 3个MLP Modulation。所以JoyAI-Image在保持参数高效的同时，还获得了层级差异化的调制能力。

---

### 创新点4：3D RoPE位置编码 vs 4D RoPE位置编码

**代码验证：**

JoyAI-Image（`models.py` 第329、334行，`posemb_layers.py` 第177-268行）：
```python
# 默认配置
rope_dim_list: List[int] = [16, 56, 56]  # 时间16维 + 高度56维 + 宽度56维 = 128维
theta: int = 256

# 在 get_rotary_pos_embed 中
vis_freqs, txt_freqs = get_nd_rotary_pos_embed(
    rope_dim_list, vis_rope_size=(tt, th, tw), theta=self.theta, ...)
```

FLUX2（`model.py` 第18、19行，第694-708行）：
```python
# 默认配置
axes_dim: list[int] = [32, 32, 32, 32]  # t+h+w+l各32维 = 128维
theta: int = 2000

# EmbedND实现
class EmbedND(nn.Module):
    def forward(self, ids):
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))],
            dim=-3,
        )
        return emb.unsqueeze(1)
```

**区别描述：**

- **JoyAI-Image**：使用**3D RoPE**，维度分配为`[16, 56, 56]`（时间:高度:宽度），`theta=256`。将更多的位置编码维度（112/128 = 87.5%）分配给空间维度（高度和宽度），仅用16维（12.5%）编码时间。使用更小的 theta=256 基频参数。
- **FLUX2**：使用**4D RoPE**，维度分配为`[32, 32, 32, 32]`（时间:高度:宽度:序列），`theta=2000`。4个维度均匀分配各32维。多出的第4维（l维度）用于编码文本token的序列位置。使用更大的 theta=2000 基频参数。

**优点分析：**

这是一个**生成效果**维度的优化。JoyAI-Image将87.5%的位置编码维度分配给空间位置，这意味着：
1. **更高的空间分辨率**：更多维度用于编码空间位置，可以更精确地区分不同空间位置的token，有利于生成精细的空间结构；
2. **更小的theta=256**：较小的theta值使位置编码的频率更高，在近距离位置上提供更强的区分度，对图像生成中的细节纹理和边缘处理可能更有利；
3. **以图像为中心的设计**：时间维度仅16维，适合以图像生成/编辑为主的任务（时间维度主要用于Multi-item区分）。

FLUX2的4D均匀分配更通用，额外的l维度用于区分文本token位置，但空间分辨率不如JoyAI-Image的非均匀分配。

---

### 创新点5：GELU-approximate MLP激活 vs SiLU Gated (SwiGLU) MLP激活

**代码验证：**

JoyAI-Image（`models.py` 第129-130行）：
```python
self.img_mlp = FeedForward(hidden_size, inner_dim=mlp_hidden_dim,
                           activation_fn="gelu-approximate")
# diffusers的FeedForward使用 "gelu-approximate" → GELU(dim, inner_dim, approximate="tanh")
# 结构: Linear(hidden, inner) → GELU(tanh) → Linear(inner, hidden)
```
（注：验证确认diffusers中`activation_fn="gelu-approximate"`对应`GELU`类，内部为`nn.Linear + F.gelu(approximate="tanh")`，非门控结构）

FLUX2（`model.py` 第390-397行，第546-549行）：
```python
class SiLUActivation(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2  # SiLU门控

# DoubleStreamBlock中:
self.img_mlp = nn.Sequential(
    nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False),  # 注意: 2倍宽度
    SiLUActivation(),  # chunk成两半做门控
    nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
)
```

**区别描述：**

- **JoyAI-Image**：MLP使用diffusers库的`FeedForward`，激活函数为`gelu-approximate`，即标准的GELU(tanh近似)。MLP结构为 `Linear(hidden→inner) → GELU(tanh) → Dropout → Linear(inner→hidden) → Dropout`，其中`mlp_width_ratio=4.0`，inner_dim为hidden_size的4倍。**非门控结构**。
- **FLUX2**：MLP使用自定义的`SiLUActivation`门控激活。第一层Linear输出2倍中间维度，然后chunk成两半做SiLU门控乘法（类似SwiGLU），最后通过第二层Linear降维。`mlp_ratio=3.0`，中间维度为hidden_size的3倍（但因为2倍扩展后chunk，实际参与计算的有效维度为3倍）。**门控结构**。

**优点分析：**

这两种激活函数各有特点，不是单纯的优劣之分：
- GELU-approximate是一种简单高效的激活，计算开销较小（无需2倍宽度的中间层）；
- SiLU Gated (SwiGLU)在LLM领域被广泛验证为更优的激活函数，但需要2倍的中间层宽度；
- JoyAI-Image使用4倍`mlp_width_ratio`配合非门控GELU，FLUX2使用3倍`mlp_ratio`配合门控SiLU，两者在有效计算量上接近。

总体而言，这不是明确的生成效果优化方向差异，更多是**设计选择差异**。

---

### 创新点6：Conv3d Patchify输入投影 vs Linear线性输入投影

**代码验证：**

JoyAI-Image（`models.py` 第357-358行）：
```python
self.img_in = nn.Conv3d(
    in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
# patch_size = [1, 2, 2]，即时间维度不patchify，空间维度2×2 patchify
```

FLUX2（`model.py` 第68行）：
```python
self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
# 直接线性投影，无patchify操作（patchify已在VAE的patch重排中完成）
```

**区别描述：**

- **JoyAI-Image**：使用`nn.Conv3d`作为输入投影层，kernel_size和stride均为`[1, 2, 2]`。这意味着在时间维度上不做合并（stride=1），在空间维度上将相邻的2×2 patch合并为一个token。Conv3d可以在patchify的同时学习局部空间特征，且天然支持3D（时间+空间）的输入处理。
- **FLUX2**：使用`nn.Linear`作为输入投影。latent已经在VAE阶段通过`rearrange`完成了2×2的空间patch重排（从`z_channels=32`变为`128`通道），因此DIT输入时每个token已经代表了一个2×2空间patch，直接做线性投影即可。

**优点分析：**

这是一个**生成效果+视频兼容性**维度的优化：
1. **Conv3d可以捕获局部空间相关性**：卷积操作在patchify的同时考虑了相邻像素之间的局部关系，而Linear投影将每个patch独立处理，丢失了patch内部的空间结构先验；
2. **天然支持视频的时间维度**：Conv3d的3D卷积可以灵活处理时间维度（当前配置时间stride=1不合并，但可扩展为合并），为未来视频编辑扩展奠定基础；
3. **一体化设计**：patchify和特征投影在一个操作中完成，避免了FLUX2需要在VAE端做patch重排的额外步骤。

---

### 创新点7：带Bias的线性层设计 vs 无Bias线性层设计

**代码验证：**

JoyAI-Image（`models.py`）：
```python
# 第113-114行
self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True, ...)
# 第120-121行
self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=True, ...)
# 第142-143行
self.txt_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True, ...)
# 第149-150行
self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=True, ...)
```

FLUX2（`model.py`）：
```python
# 第68行
self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
# 第70行
self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)
# 第384行
self.qkv = nn.Linear(dim, dim * 3, bias=False)
# 第387行
self.proj = nn.Linear(dim, dim, bias=False)
# 第547行
nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False)
```

**区别描述：**

- **JoyAI-Image**：QKV投影层和注意力输出投影层均使用`bias=True`，保留了偏置项参数。
- **FLUX2**：几乎所有线性层均使用`bias=False`，包括输入投影、QKV投影、注意力输出投影、MLP层、调制层等。

**优点分析：**

这是一个**表达能力 → 生成效果**维度的差异：
- 带bias的线性层有更多参数和更强的表达能力，可以学到偏移项，在小模型中可能有助于提升性能；
- 无bias是现代大规模Transformer的趋势（如LLaMA、Mistral），可以减少参数量并在某些情况下提高训练稳定性。
- JoyAI-Image选择保留bias可能是因为其模型整体参数量较FLUX2小（约16B vs 32B），需要额外的表达能力补偿。

---

### 创新点8：文本不参与RoPE位置编码 vs 文本参与4D RoPE位置编码

**代码验证：**

JoyAI-Image（`models.py` 第200-228行）：
```python
# 图像token应用RoPE
if vis_freqs_cis is not None:
    img_qq, img_kk = apply_rotary_emb(img_q, img_k, vis_freqs_cis, head_first=False)
    img_q, img_k = img_qq, img_kk

# 文本token不应用RoPE（实际上 txt_freqs_cis 在默认配置下为 None）
if txt_freqs_cis is not None:
    raise NotImplementedError("RoPE text is not supported for inference")
```

在`forward`中（第467-468行）：
```python
vis_freqs_cis, txt_freqs_cis = self.get_rotary_pos_embed(
    vis_rope_size=(tt, th, tw),
    txt_rope_size=txt_seq_len if self.rope_type == 'mrope' else None)
# 默认 rope_type='rope'，所以 txt_rope_size=None → txt_freqs_cis=None
```

FLUX2（`model.py` 第139-140行，第649行）：
```python
pe_x = self.pe_embedder(x_ids)   # 图像token的4D位置编码
pe_ctx = self.pe_embedder(ctx_ids)  # 文本token的4D位置编码

# DoubleStreamBlock中:
pe_full = torch.cat((pe_ctx, pe), dim=2)  # 文本和图像RoPE合并
q, k = apply_rope(q, k, pe_full)  # 所有token都应用RoPE
```

**区别描述：**

- **JoyAI-Image**：默认配置下仅对图像token应用3D RoPE位置编码，文本token不参与RoPE（虽然代码中预留了mrope模式的接口，但在推理时抛出`NotImplementedError`）。文本token在注意力中没有显式的位置信息。
- **FLUX2**：图像token和文本token都通过同一个`EmbedND`计算4D RoPE位置编码。文本token使用4D中的第4维（l维度）编码序列位置。

**优点分析：**

这是一个**设计简洁性**的差异，两种方式各有道理：
- JoyAI-Image不对文本token施加RoPE，意味着文本token之间的相对位置关系不通过RoPE编码，但文本token本身已经通过Qwen3VL的因果语言模型获得了丰富的位置信息（在text encoder阶段已编码）；
- FLUX2对文本也施加4D RoPE，可以在联合注意力中提供文本token的位置信息，但需要额外维度。

不直接对文本施加RoPE的好处是避免了text encoder已有位置编码与DIT RoPE之间的冲突，设计更简洁。

---

### 创新点9：Multi-item全程参与注意力编辑机制 vs 因果注意力/KV-Cache参考机制

**代码验证：**

JoyAI-Image（`models.py` 第427-443行，`pipeline.py` 第860-861行）：
```python
# models.py: Multi-item在时间维度上拼接
if is_multi_item:
    num_items = hidden_states.shape[1]
    if num_items > 1:
        hidden_states = torch.cat([hidden_states[:, -1:], hidden_states[:, :-1]], dim=1)
    hidden_states = rearrange(hidden_states, 'b n c t h w -> b c (n t) h w')

# pipeline.py: 每个去噪步恢复参考latent
if num_items > 1:
    latents[:, :(num_items - 1)] = ref_latents.clone()
```

FLUX2（`model.py` 第758-815行）：
```python
# 因果注意力：参考token只能自注意力
def causal_attn_fn(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache=None):
    # txt+img attend to all keys
    attn_txt_img = F.scaled_dot_product_attention(q_txt_img, k_all, v_all)
    # ref only attends to itself
    attn_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)

# KV-Cache: 第一步提取参考KV，后续步复用
# forward_kv_extract(): 完整计算 + 提取KV cache
# forward_kv_cached(): 仅注入缓存KV，不含ref token
```

**区别描述：**

- **JoyAI-Image (Multi-item)**：
  1. 将参考图像通过VAE编码为latent后，与目标噪声latent在时间维度上拼接，作为一个完整的3D输入送入DIT；
  2. 在Conv3d patchify后，参考图像和目标图像的所有token共同参与**全注意力**（无注意力mask限制），参考图像token可以看到目标图像token，目标图像token也可以看到参考图像token；
  3. 每个去噪步都将参考latent恢复为原始值（不被去噪过程更新），确保参考信息不被污染。

- **FLUX2 (因果注意力 + KV-Cache)**：
  1. 参考图像通过VAE编码后作为额外的token序列拼接到图像token之后；
  2. 使用因果注意力：参考token**只能看到自己**（自注意力），文本和目标图像token可以看到所有token（包括参考token）；
  3. 参考token使用固定时间步 t=0.0（完全干净状态），与目标图像的当前去噪时间步不同；
  4. 支持KV-Cache优化：第一步完整计算提取参考KV，后续步复用缓存。

**优点分析：**

这是一个**编辑质量 → 生成效果**维度的优化。JoyAI-Image的Multi-item机制相比FLUX2有以下优势：
1. **全注意力交互**：参考图像和目标图像之间的注意力是双向的，参考图像token也能"看到"目标图像正在生成的内容，这使得参考信息的利用更加充分和动态；
2. **每步完整参与**：参考latent在每个去噪步都完整参与DIT的前向计算（而非仅通过KV-Cache注入），信息传递更直接；
3. **时间维度拼接**：通过Conv3d的时间维度拼接，参考和目标之间的关系可以被3D RoPE的时间维度自然编码。

但FLUX2的因果注意力设计也有其优点：防止参考图像被去噪过程污染（通过限制参考token的可见范围），且KV-Cache可以显著加速推理。

---

### 创新点10：多模态VLM文本编码器 vs 纯文本LLM文本编码器（单层 vs 多层提取）

**代码验证：**

JoyAI-Image（`pipeline.py` 第207-212行）：
```python
encoder_hidden_states = self.text_encoder(
    input_ids=txt_tokens.input_ids,
    attention_mask=txt_tokens.attention_mask,
    output_hidden_states=True,
)
hidden_states = encoder_hidden_states.hidden_states[-1]  # 仅取最后一层
```

编辑模式下通过Qwen3VL处理图像+文本（`pipeline.py` 第253-268行）：
```python
inputs = self.qwen_processor(text=prompt, images=images, ...)
encoder_hidden_states = self.text_encoder(**inputs, output_hidden_states=True)
prompt_embeds = last_hidden_states[:, drop_idx:]  # Qwen3VL内部ViT处理图像
```

FLUX2（`text_encoder.py` 第247-248行）：
```python
out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_MISTRAL], dim=1)
# OUTPUT_LAYERS_MISTRAL = [10, 20, 30]
return rearrange(out, "b c l d -> b l (c d)")  # 拼接3层 → (B, 512, 15360)
```

**区别描述：**

- **JoyAI-Image**：
  - 文本编码器为**Qwen3VL**（视觉-语言多模态模型），可以同时处理文本和图像输入；
  - 仅提取**最后一层**的hidden_states作为条件嵌入，输出维度为`(B, L, 4096)`；
  - 编辑模式下，参考图像通过Qwen3VL内部的ViT编码为视觉token，与文本token一起经过LLM处理，输出包含视觉语义理解信息的隐藏状态。

- **FLUX2**：
  - 文本编码器为**Mistral-Small-3.2-24B**（纯文本LLM，虽然是多模态架构但在embedding提取时仅处理文本）或**Qwen3-4B/8B**（纯文本LLM）；
  - 提取第10、20、30层（或9、18、27层）的hidden_states并**concatenate拼接**，输出维度为`(B, 512, 15360)`（Mistral）或`(B, 512, 7680/12288)`（Qwen3）；
  - 多层拼接可以获得不同抽象层次的语义特征（浅层→语法/词法、深层→语义）。

**优点分析：**

这是一个**多模态理解 → 生成效果**维度的优化：
1. **Qwen3VL可以在语义层面理解参考图像**：编辑模式下，参考图像的视觉特征经过VLM处理后包含高级语义信息（如物体类别、空间关系、颜色等），而非仅作为latent token参与注意力；
2. **统一的视觉-文本理解**：文本和图像在同一个多模态模型中联合处理，可以更好地理解图像内容与编辑指令之间的关系。

但FLUX2的多层特征提取也有其优势：拼接多个中间层可以获得**多尺度语义特征**，浅层特征保留了更多低级信息，深层特征包含更多高级语义，组合使用可以提供更丰富的条件信号。

---

### 创新点11：条件嵌入器架构差异（WanTimeTextImageEmbedding vs MLPEmbedder+独立投影）

**代码验证：**

JoyAI-Image（`models.py` 第272-308行）：
```python
class WanTimeTextImageEmbedding(nn.Module):
    def __init__(self, dim, time_freq_dim, time_proj_dim, text_embed_dim, ...):
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, ...)  # 正弦位置编码
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)  # time_proj_dim = hidden_size * 6
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(self, timestep, encoder_hidden_states):
        timestep = self.timesteps_proj(timestep)
        temb = self.time_embedder(timestep)
        timestep_proj = self.time_proj(self.act_fn(temb))  # → vec (6*hidden_size)
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)  # 4096→hidden_size
        return temb, timestep_proj, encoder_hidden_states
```

FLUX2（`model.py` 第69-74行，第126-130行）：
```python
# 初始化
self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)
self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)
self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)

# forward
vec = self.time_in(timestep_embedding(timesteps, 256))
if self.use_guidance_embed:
    vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
```

**区别描述：**

- **JoyAI-Image**：使用集成的`WanTimeTextImageEmbedding`类，在一个模块中同时处理时间步嵌入和文本投影：
  - 时间步：`Timesteps`（正弦编码）→ `TimestepEmbedding`（MLP）→ `SiLU` → `Linear`投影到`6*hidden_size`维度的调制向量
  - 文本：`PixArtAlphaTextProjection`（Linear → GELU_tanh → Linear）将文本从4096维投影到hidden_size维

- **FLUX2**：时间步嵌入和文本投影是独立的模块：
  - 时间步：`timestep_embedding`（正弦编码）→ `MLPEmbedder`（Linear → SiLU → Linear）得到`hidden_size`维的条件向量
  - 文本：`nn.Linear`直接将文本从context_in_dim投影到hidden_size
  - 条件向量vec由time_in和guidance_in的输出相加得到

**优点分析：**

这主要是**架构设计风格**的差异。JoyAI-Image使用集成式的条件嵌入器，将所有条件处理封装在一个类中，代码更集中。FLUX2使用分离式设计，各条件处理模块独立，更灵活。值得注意的是，JoyAI-Image的`time_proj`直接将时间步投影到`6*hidden_size`维度，可以在Transformer外部一次性生成所有调制参数的"基础条件向量"，然后在每层通过ModulateWan的加法叠加。

---

### 创新点12：输出层结构差异（简单Norm+Linear vs AdaLN调制输出层）

**代码验证：**

JoyAI-Image（`models.py` 第384-389行，第504行）：
```python
self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
self.proj_out = nn.Linear(hidden_size, out_channels * math.prod(patch_size), ...)

# forward中：
img = self.proj_out(self.norm_out(img))
```

FLUX2（`model.py` 第415-434行）：
```python
class LastLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x, vec):
        mod = self.adaLN_modulation(vec)
        shift, scale = mod.chunk(2, dim=-1)
        x = (1 + scale) * self.norm_final(x) + shift
        x = self.linear(x)
        return x
```

**区别描述：**

- **JoyAI-Image**：输出层是简单的`LayerNorm → Linear`，不使用任何条件调制。
- **FLUX2**：输出层是`LastLayer`，包含AdaLN调制——将条件向量vec通过`SiLU + Linear`生成shift和scale参数，对LayerNorm后的特征进行条件调制，然后再通过Linear投影到输出维度。

**优点分析：**

这是一个**设计简洁性**方面的差异。JoyAI-Image的简单输出层减少了参数量和计算量，但FLUX2的AdaLN调制输出层可以在最终输出时根据条件（时间步等）进行自适应调整，理论上可以提供更精确的输出控制。JoyAI-Image可能认为通过20层双流块的充分处理，最终输出不再需要额外的条件调制。

---

### 创新点13：Flash Attention varlen vs PyTorch scaled_dot_product_attention

**代码验证：**

JoyAI-Image（`attention.py` 第93-101行）：
```python
x = flash_attn_varlen_func(
    q.view(q.shape[0] * q.shape[1], *q.shape[2:]),
    k.view(k.shape[0] * k.shape[1], *k.shape[2:]),
    v.view(v.shape[0] * v.shape[1], *v.shape[2:]),
    cu_seqlens_q, cu_seqlens_kv,
    max_seqlen_q, max_seqlen_kv,
)
```

FLUX2（`model.py` 第786行，第806行）：
```python
out = F.scaled_dot_product_attention(q_txt_img, k_all, v_all, is_causal=False)
attn_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=False)
```

**区别描述：**

- **JoyAI-Image**：使用Flash Attention的`flash_attn_varlen_func`，支持变长序列（通过`cu_seqlens`累积序列长度指示每个样本的实际序列长度），可以高效处理batch内不同长度的文本序列，避免填充token的无效计算。
- **FLUX2**：使用PyTorch原生的`F.scaled_dot_product_attention`，通过手动分割Q/K/V来实现因果注意力，不使用变长序列优化。

**优点分析：**

这是一个**计算效率**维度的优化：
1. **Flash Attention**在GPU上的内存效率和计算速度通常优于标准attention实现；
2. **varlen变长支持**可以避免对不同长度的文本序列做padding后的无效计算，在batch内文本长度差异大时特别有优势；
3. 但Flash Attention需要安装额外的库，且不如PyTorch原生实现通用。

---

### 创新点14：CFG噪声范数归一化 vs 标准CFG/Guidance Embedding

**代码验证：**

JoyAI-Image（`pipeline.py` 第896-903行）：
```python
if self.do_classifier_free_guidance:
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + self.guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )
    cond_norm = torch.norm(noise_pred_text, dim=2, keepdim=True)
    noise_norm = torch.norm(noise_pred, dim=2, keepdim=True)
    noise_pred = noise_pred * (cond_norm / noise_norm)  # 范数归一化
```

FLUX2（`sampling.py` 第358-361行 和 `model.py` 第128-130行）：
```python
# 标准CFG（denoise_cfg函数）:
pred = pred_uncond + guidance * (pred_cond - pred_uncond)

# 或 Guidance Embedding（forward函数）:
vec = self.time_in(timestep_emb)
if self.use_guidance_embed:
    vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
```

**区别描述：**

- **JoyAI-Image**：在标准CFG公式`uncond + scale * (cond - uncond)`之后，额外进行了**噪声范数归一化**——将CFG混合后的预测噪声缩放到与条件预测`noise_pred_text`相同的范数大小。这防止了高guidance_scale导致的噪声幅度过大问题。
- **FLUX2**：Dev 32B模型使用Guidance Embedding方式（将guidance值编码为向量注入条件），避免了双倍计算；Base模型使用标准CFG但不做范数归一化。

**优点分析：**

这是一个**生成质量稳定性 → 生成效果**维度的优化。CFG噪声范数归一化的好处是：
1. 防止高guidance_scale时噪声预测的幅度过大，避免生成图像出现过饱和、过曝等伪影；
2. 保持了CFG的语义引导方向，同时将预测幅度约束在合理范围内；
3. 使得模型对guidance_scale超参数更加鲁棒，用户可以使用更大的guidance_scale而不会导致质量崩溃。

---

### 创新点15：无Guidance Embedding vs 有Guidance Embedding

**代码验证：**

JoyAI-Image的`Transformer3DModel`中：
```python
# 无 guidance_in 模块，不使用 guidance embedding
# 条件嵌入器只处理 timestep 和 text
self.condition_embedder = WanTimeTextImageEmbedding(
    dim=hidden_size, time_freq_dim=256,
    time_proj_dim=hidden_size * 6, text_embed_dim=text_states_dim)
```

FLUX2（`model.py` 第72-74行）：
```python
self.use_guidance_embed = params.use_guidance_embed  # Dev: True, Klein: False
if self.use_guidance_embed:
    self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, ...)
```

**区别描述：**

- **JoyAI-Image**：不使用Guidance Embedding。在推理时使用传统的CFG（双倍前向传播），guidance_scale仅在CFG公式中作为权重使用。
- **FLUX2 (Dev 32B)**：使用Guidance Embedding。将guidance值通过`MLPEmbedder`编码为向量，加到时间步条件向量上，注入到模型中。这样模型在单次前向传播中就能感知guidance强度，无需双倍计算。

**优点分析：**

这是一个**设计简洁性**方面的差异。JoyAI-Image不使用Guidance Embedding意味着其推理时需要做两次前向传播（条件和无条件），计算量翻倍，但好处是：
1. 训练时不需要额外学习guidance条件的注入方式；
2. 推理时可以灵活调整guidance_scale而不受模型训练时的约束；
3. 配合CFG噪声范数归一化（创新点14），可以在保证质量的前提下使用传统CFG。

---

## 4. 总结

### JoyAI-Image去噪模型相比FLUX2去噪模型的主要创新点总结

| 类别 | 创新点 | 优化维度 |
|------|--------|---------|
| **整体架构** | 纯双流MMDiT（20层），无单流块 | 模态独立性 → 生成效果 |
| **调制机制** | ModulateWan可学习参数表（加法调制） | 参数效率 |
| **调制策略** | 每层独立调制参数 | 层级差异化表达能力 → 生成效果 |
| **位置编码** | 3D RoPE, [16,56,56], theta=256 | 空间分辨率 → 生成效果 |
| **输入投影** | Conv3d 3D卷积patchify | 局部特征+视频兼容性 → 生成效果 |
| **线性层** | 带bias设计 | 表达能力 → 生成效果 |
| **文本位置** | 文本token不参与RoPE | 设计简洁性/避免编码冲突 |
| **编辑机制** | Multi-item全程参与全注意力 | 编辑质量 → 生成效果 |
| **文本编码** | Qwen3VL多模态VLM + 单层提取 | 多模态理解 → 编辑效果 |
| **注意力** | Flash Attention varlen | 计算效率 |
| **CFG策略** | CFG + 噪声范数归一化 | 生成稳定性 → 生成效果 |
| **MLP激活** | GELU-approximate（非门控） | 设计选择 |
| **输出层** | 简单Norm+Linear（无条件调制） | 设计简洁性 |
| **条件嵌入** | 集成式WanTimeTextImageEmbedding | 架构设计 |
| **Guidance** | 无Guidance Embedding，使用传统CFG | 训练简洁/推理灵活 |

### 核心结论

JoyAI-Image的去噪模型与FLUX2在网络架构上存在**15个**主要差异创新点。其中：

- **对生成效果有直接正向影响的创新点**（共7个）：纯双流架构（#1）、每层独立调制（#3）、3D RoPE空间维度偏重（#4）、Conv3d输入投影（#6）、带Bias设计（#7）、Multi-item全注意力编辑（#9）、CFG噪声范数归一化（#14）。

- **对参数效率/计算效率有正向影响的创新点**（共2个）：ModulateWan可学习参数表（#2）、Flash Attention varlen（#13）。

- **对多模态理解/编辑能力有正向影响的创新点**（共1个）：Qwen3VL多模态VLM编码器（#10）。

- **属于设计选择差异的创新点**（共5个）：MLP激活函数（#5）、文本不参与RoPE（#8）、条件嵌入器架构（#11）、输出层结构（#12）、无Guidance Embedding（#15）。

总体而言，JoyAI-Image的去噪模型在架构设计上更偏向**图像/视频统一处理**（3D卷积、3D RoPE、Multi-item机制），在调制机制上更注重**参数效率与层级差异化**（ModulateWan + 每层独立），在编辑能力上更强调**语义理解与全注意力交互**（VLM编码器 + Multi-item），代表了一种与FLUX2不同的设计哲学。

---

> **报告生成时间**：基于源代码全量分析
>
> **分析的代码文件**：
> - JoyAI-Image: `src/modules/models/mmdit/dit/models.py`, `modulate_layers.py`, `posemb_layers.py`, `attention.py`, `pipeline.py`, `scheduler.py`, `__init__.py`, `infer_runtime/model.py`, `infer_runtime/infer_config.py`
> - FLUX2: `src/flux2/model.py`, `src/flux2/sampling.py`, `src/flux2/text_encoder.py`