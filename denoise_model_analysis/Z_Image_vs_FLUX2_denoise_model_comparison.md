# Z-Image vs FLUX2 去噪模型架构对比分析

> 本报告基于 `/root/code/Z-Image/src/zimage/transformer.py` 和 `/root/code/flux2/src/flux2/model.py` 两个代码仓库中的去噪模型（Transformer/DIT）部分的源代码进行逐一对比分析。所有结论均经过代码逻辑复核验证。
>
> 参考了 `analysis/Z_Image_universal_image_edit_model_analysis.md` 和 `analysis/FLUX2_model_analysis.md` 中的分析内容，但每条结论均已与源代码重新核实。

---

## 目录

- [Z-Image vs FLUX2 去噪模型架构对比分析](#z-image-vs-flux2-去噪模型架构对比分析)
  - [目录](#目录)
  - [1. 整体架构对比概览](#1-整体架构对比概览)
  - [整体架构创新点](#整体架构创新点)
    - [创新点1：纯单流架构 + Noise Refiner / Context Refiner 前置处理](#创新点1纯单流架构--noise-refiner--context-refiner-前置处理)
      - [FLUX2 的做法](#flux2-的做法)
      - [Z-Image 的做法](#z-image-的做法)
      - [区别与优点分析](#区别与优点分析)
    - [创新点2：独立的 Noise Refiner 模块（噪声预处理）](#创新点2独立的-noise-refiner-模块噪声预处理)
      - [FLUX2 的做法](#flux2-的做法-1)
      - [Z-Image 的做法](#z-image-的做法-1)
      - [区别与优点分析](#区别与优点分析-1)
    - [创新点3：独立的 Context Refiner 模块（文本预处理）](#创新点3独立的-context-refiner-模块文本预处理)
      - [FLUX2 的做法](#flux2-的做法-2)
      - [Z-Image 的做法](#z-image-的做法-2)
      - [区别与优点分析](#区别与优点分析-2)
  - [Attention 组件创新点](#attention-组件创新点)
    - [创新点4：分离式 Q/K/V 投影替代合并式 QKV 投影](#创新点4分离式-qkv-投影替代合并式-qkv-投影)
      - [FLUX2 的做法](#flux2-的做法-3)
      - [Z-Image 的做法](#z-image-的做法-3)
      - [区别与优点分析](#区别与优点分析-3)
    - [创新点5：支持 GQA（Grouped Query Attention）](#创新点5支持-gqagrouped-query-attention)
      - [FLUX2 的做法](#flux2-的做法-4)
      - [Z-Image 的做法](#z-image-的做法-4)
      - [区别与优点分析](#区别与优点分析-4)
  - [FFN 组件创新点](#ffn-组件创新点)
    - [创新点6：SwiGLU FFN（三矩阵门控）替代双矩阵 SiLU Gated MLP](#创新点6swiglu-ffn三矩阵门控替代双矩阵-silu-gated-mlp)
      - [FLUX2 的做法](#flux2-的做法-5)
      - [Z-Image 的做法](#z-image-的做法-5)
      - [区别与优点分析](#区别与优点分析-5)
  - [归一化策略创新点](#归一化策略创新点)
    - [创新点7：Pre-RMSNorm + Post-RMSNorm 双重归一化替代 AdaLN Pre-Norm 单归一化](#创新点7pre-rmsnorm--post-rmsnorm-双重归一化替代-adaln-pre-norm-单归一化)
      - [FLUX2 的做法](#flux2-的做法-6)
      - [Z-Image 的做法](#z-image-的做法-6)
      - [区别与优点分析](#区别与优点分析-6)
  - [AdaLN 调制创新点](#adaln-调制创新点)
    - [创新点8：AdaLN 调制参数精简为 4 个，去掉 shift](#创新点8adaln-调制参数精简为-4-个去掉-shift)
      - [FLUX2 的做法](#flux2-的做法-7)
      - [Z-Image 的做法](#z-image-的做法-7)
      - [区别与优点分析](#区别与优点分析-7)
    - [创新点9：Gate 使用 Tanh 激活替代线性（无约束）Gate](#创新点9gate-使用-tanh-激活替代线性无约束gate)
      - [FLUX2 的做法](#flux2-的做法-8)
      - [Z-Image 的做法](#z-image-的做法-8)
      - [区别与优点分析](#区别与优点分析-8)
    - [创新点10：AdaLN Embedding 维度独立于 hidden\_size（ADALN\_EMBED\_DIM=256）](#创新点10adaln-embedding-维度独立于-hidden_sizeadaln_embed_dim256)
      - [FLUX2 的做法](#flux2-的做法-9)
      - [Z-Image 的做法](#z-image-的做法-9)
      - [区别与优点分析](#区别与优点分析-9)
    - [创新点11：每层独立 Modulation 替代全局共享 Modulation](#创新点11每层独立-modulation-替代全局共享-modulation)
      - [FLUX2 的做法](#flux2-的做法-10)
      - [Z-Image 的做法](#z-image-的做法-10)
      - [区别与优点分析](#区别与优点分析-10)
  - [位置编码创新点](#位置编码创新点)
    - [创新点12：3D RoPE（t, h, w）替代 4D RoPE（t, h, w, l）](#创新点123d-ropet-h-w替代-4d-ropet-h-w-l)
      - [FLUX2 的做法](#flux2-的做法-11)
      - [Z-Image 的做法](#z-image-的做法-11)
      - [区别与优点分析](#区别与优点分析-11)
    - [创新点13：RoPE theta=256 替代 theta=2000](#创新点13rope-theta256-替代-theta2000)
      - [FLUX2 的做法](#flux2-的做法-12)
      - [Z-Image 的做法](#z-image-的做法-12)
      - [区别与优点分析](#区别与优点分析-12)
    - [创新点14：复数乘法实现 RoPE 替代矩阵旋转实现 RoPE](#创新点14复数乘法实现-rope-替代矩阵旋转实现-rope)
      - [FLUX2 的做法](#flux2-的做法-13)
      - [Z-Image 的做法](#z-image-的做法-13)
      - [区别与优点分析](#区别与优点分析-13)
  - [输出层创新点](#输出层创新点)
    - [创新点15：FinalLayer 仅使用 scale（无 shift）调制](#创新点15finallayer-仅使用-scale无-shift调制)
      - [FLUX2 的做法](#flux2-的做法-14)
      - [Z-Image 的做法](#z-image-的做法-14)
      - [区别与优点分析](#区别与优点分析-14)
  - [序列处理创新点](#序列处理创新点)
    - [创新点16：支持变长序列 Padding + Attention Mask](#创新点16支持变长序列-padding--attention-mask)
      - [FLUX2 的做法](#flux2-的做法-15)
      - [Z-Image 的做法](#z-image-的做法-15)
      - [区别与优点分析](#区别与优点分析-15)
    - [创新点17：Learnable Pad Token（可学习的填充令牌）](#创新点17learnable-pad-token可学习的填充令牌)
      - [FLUX2 的做法](#flux2-的做法-16)
      - [Z-Image 的做法](#z-image-的做法-16)
      - [区别与优点分析](#区别与优点分析-16)
  - [输入/输出适配创新点](#输入输出适配创新点)
    - [创新点18：多 Patch Size 支持（动态分辨率适配）](#创新点18多-patch-size-支持动态分辨率适配)
      - [FLUX2 的做法](#flux2-的做法-17)
      - [Z-Image 的做法](#z-image-的做法-17)
      - [区别与优点分析](#区别与优点分析-17)
    - [创新点19：使用传统 SD-style VAE（8x下采样, 16通道）替代 FLUX2 VAE（16x下采样, 128通道）](#创新点19使用传统-sd-style-vae8x下采样-16通道替代-flux2-vae16x下采样-128通道)
      - [FLUX2 的做法](#flux2-的做法-18)
      - [Z-Image 的做法](#z-image-的做法-18)
      - [区别与优点分析](#区别与优点分析-18)
  - [条件注入创新点](#条件注入创新点)
    - [创新点20：Timestep 先乘以 t\_scale=1000 再送入 Embedder](#创新点20timestep-先乘以-t_scale1000-再送入-embedder)
      - [FLUX2 的做法](#flux2-的做法-19)
      - [Z-Image 的做法](#z-image-的做法-19)
      - [区别与优点分析](#区别与优点分析-19)
    - [创新点21：文本条件投影使用 RMSNorm + Linear 替代单 Linear](#创新点21文本条件投影使用-rmsnorm--linear-替代单-linear)
      - [FLUX2 的做法](#flux2-的做法-20)
      - [Z-Image 的做法](#z-image-的做法-20)
      - [区别与优点分析](#区别与优点分析-20)
  - [参数策略创新点](#参数策略创新点)
    - [创新点22：Linear 层部分使用 bias=True（区分条件路径与计算路径）](#创新点22linear-层部分使用-biastrue区分条件路径与计算路径)
      - [FLUX2 的做法](#flux2-的做法-21)
      - [Z-Image 的做法](#z-image-的做法-21)
      - [区别与优点分析](#区别与优点分析-21)
  - [工程优化创新点](#工程优化创新点)
    - [创新点23：丰富的 Attention Backend 调度系统](#创新点23丰富的-attention-backend-调度系统)
      - [FLUX2 的做法](#flux2-的做法-22)
      - [Z-Image 的做法](#z-image-的做法-22)
      - [区别与优点分析](#区别与优点分析-22)
  - [推理策略创新点](#推理策略创新点)
    - [创新点24：CFG Truncation 机制](#创新点24cfg-truncation-机制)
      - [FLUX2 的做法](#flux2-的做法-23)
      - [Z-Image 的做法](#z-image-的做法-23)
      - [区别与优点分析](#区别与优点分析-23)
    - [创新点25：CFG Normalization 机制](#创新点25cfg-normalization-机制)
      - [FLUX2 的做法](#flux2-的做法-24)
      - [Z-Image 的做法](#z-image-的做法-24)
      - [区别与优点分析](#区别与优点分析-24)
  - [总结表](#总结表)
  - [总体设计哲学总结](#总体设计哲学总结)

---

## 1. 整体架构对比概览

| 特征 | Z-Image (`ZImageTransformer2DModel`) | FLUX2 (`Flux2`) |
|------|--------------------------------------|-----------------|
| **架构类型** | **纯单流** + Noise/Context Refiner 前置 | 双流（DoubleStreamBlock）+ 单流（SingleStreamBlock）|
| **主干层** | `ZImageTransformerBlock × 30` | `DoubleStreamBlock × 8` → `SingleStreamBlock × 48` |
| **前置处理** | `noise_refiner × 2` + `context_refiner × 2` | 无 |
| **hidden_size** | 3840 | 6144 (dev) / 4096 (klein-9B) / 3072 (klein-4B) |
| **num_heads** | 30 | 48 / 32 / 24 |
| **head_dim** | 128 | 128 |
| **in_channels** | 16 | 128 |
| **FFN 类型** | SwiGLU（三矩阵，hidden_dim=int(dim/3*8)） | SiLU Gated（双矩阵 + chunk，mlp_ratio=3.0） |
| **Attention** | 分离 Q/K/V + 支持 GQA | 合并 QKV |
| **Norm** | RMSNorm (pre + post，带可学习参数) | LayerNorm (pre only，无可学习参数) |
| **Modulation** | 每层独立, 4 参数(scale_msa,gate_msa,scale_mlp,gate_mlp) | 全局共享, 6 参数(shift,scale,gate ×2) |
| **Gate 激活** | Tanh（约束到 [-1, 1]） | 无约束（线性） |
| **RoPE** | 3D (t,h,w), theta=256, axes_dims=[32,48,48], 复数乘法 | 4D (t,h,w,l), theta=2000, axes_dim=[32,32,32,32], 矩阵旋转 |
| **VAE** | SD-style (8x下采样, 16ch latent, patch_size=2) | FLUX2-style (16x下采样, 128ch latent) |
| **序列处理** | 变长序列 + padding + attention mask + 可学习 pad token | 固定长度, 无 mask |

---

## 整体架构创新点

### 创新点1：纯单流架构 + Noise Refiner / Context Refiner 前置处理

#### FLUX2 的做法
FLUX2 使用**双流+单流**混合架构：先经过 8 层 `DoubleStreamBlock`（图像和文本分别有独立的 Norm/QKV/MLP，但在 Attention 计算时进行 Joint Attention），再将 txt 和 img 拼接后经过 48 层 `SingleStreamBlock`。

```python
# flux2/model.py Flux2.forward() 第142-165行
for block in self.double_blocks:
    img, txt, _ = block.forward_kv_extract(img, txt, pe_x, pe_ctx,
                                            double_block_mod_img, double_block_mod_txt, num_ref_tokens=0)
img = torch.cat((txt, img), dim=1)
pe = torch.cat((pe_ctx, pe_x), dim=2)
for block in self.single_blocks:
    img, _ = block.forward_kv_extract(img, pe, single_block_mod, num_txt_tokens, num_ref_tokens=0)
img = img[:, num_txt_tokens:, ...]
img = self.final_layer(img, vec)
```

#### Z-Image 的做法
Z-Image 使用**纯单流**架构，但在主干层之前引入了两个前置 Refiner 模块：
1. **Noise Refiner**（2层）：仅对**噪声图像 tokens** 进行 self-attention + AdaLN 调制预处理
2. **Context Refiner**（2层）：仅对**文本 tokens** 进行 self-attention 预处理（**无** AdaLN 调制）
3. 然后将处理后的噪声 tokens 和文本 tokens **拼接**，送入 30 层统一的 `ZImageTransformerBlock`

```python
# Z-Image/transformer.py ZImageTransformer2DModel.forward() 第521-565行
# 1. Noise Refiner: 仅处理 x (噪声), 带 adaln_input
for layer in self.noise_refiner:
    x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

# 2. Context Refiner: 仅处理 cap_feats (文本), 注意无 adaln_input
for layer in self.context_refiner:
    cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

# 3. 拼接后送入主干
unified = []
for i in range(bsz):
    x_len = x_item_seqlens[i]
    cap_len = cap_item_seqlens[i]
    unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
    unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))

for layer in self.layers:
    unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)
```

#### 区别与优点分析
- **生成效果优化**：Noise Refiner 让噪声 tokens 在进入主干之前就经过了带时间步调制的 self-attention 预处理，使其内部表征更加结构化；Context Refiner 让文本 tokens 在不受时间步影响的情况下先做独立的 self-attention 来增强文本表征。这种"先独立精炼，再联合交互"的设计使得主干层的每一层都可以专注于图文联合交互，而不需要像 FLUX2 双流块那样在前 8 层中同时承担"独立特征精炼"和"跨模态交互"两个任务。
- **计算效率优化**：Refiner 仅各 2 层，且分别处理较短的序列（噪声或文本），计算成本远低于 FLUX2 的 8 层双流块。双流块中图像和文本各自维护独立的 QKV 投影和 MLP，参数量约为单流块的 2 倍，而 Z-Image 的 Refiner 使用与主干完全相同的 block 结构，避免了参数冗余。

---

### 创新点2：独立的 Noise Refiner 模块（噪声预处理）

#### FLUX2 的做法
FLUX2 的噪声图像 tokens 经过 `img_in` 线性投影后直接进入双流块，没有任何独立预处理。

```python
# flux2/model.py Flux2.forward() 第136行
img = self.img_in(x)
# 直接进入 double_blocks
```

#### Z-Image 的做法
Z-Image 引入了 2 层 `noise_refiner`，每层是带 `modulation=True` 的 `ZImageTransformerBlock`：

```python
# Z-Image/transformer.py 第308-313行
self.noise_refiner = nn.ModuleList([
    ZImageTransformerBlock(1000 + layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True)
    for layer_id in range(n_refiner_layers)  # n_refiner_layers=2
])
```

注意 `layer_id` 以 1000 开始，与主干层的 layer_id（0~29）不冲突，确保不同的身份标识。

在前向传播中，noise_refiner 接收 `adaln_input`（时间步 embedding）：
```python
# 第521-522行
for layer in self.noise_refiner:
    x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)
```

#### 区别与优点分析
- **生成效果优化**：Noise Refiner 带有 AdaLN 调制（使用时间步信息），这意味着噪声 tokens 在进入主干之前就已经获得了时间步感知的自注意力处理。这可以帮助模型在早期阶段就理解当前去噪阶段的特征，从而在主干层中更高效地与文本条件融合。噪声图像在不同去噪阶段（早期高噪声 vs 后期低噪声）具有非常不同的统计特性，Noise Refiner 可以针对性地调整特征表示。

---

### 创新点3：独立的 Context Refiner 模块（文本预处理）

#### FLUX2 的做法
FLUX2 的文本 tokens 经过 `txt_in` 线性投影后直接进入双流块，没有独立预处理。

```python
# flux2/model.py Flux2.forward() 第137行
txt = self.txt_in(ctx)
# 直接进入 double_blocks
```

#### Z-Image 的做法
Z-Image 引入了 2 层 `context_refiner`，每层是 `modulation=False` 的 `ZImageTransformerBlock`：

```python
# Z-Image/transformer.py 第315-320行
self.context_refiner = nn.ModuleList([
    ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=False)
    for layer_id in range(n_refiner_layers)  # n_refiner_layers=2
])
```

关键：`modulation=False` 意味着 Context Refiner **不使用时间步调制**，前向传播为标准的 self-attention + FFN：

```python
# Z-Image/transformer.py 第193-200行 (modulation=False 分支)
attn_out = self.attention(
    self.attention_norm1(x),
    attention_mask=attn_mask,
    freqs_cis=freqs_cis,
)
x = x + self.attention_norm2(attn_out)
x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
```

在前向传播中，context_refiner 不接收 adaln_input：
```python
# 第545-546行
for layer in self.context_refiner:
    cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)  # 注意没有 adaln_input 参数
```

#### 区别与优点分析
- **生成效果优化**：Context Refiner 不使用时间步调制是一个精妙的设计。文本语义本身不应该随去噪时间步变化（文本含义在整个去噪过程中保持不变），因此独立的无调制 self-attention 可以让文本 tokens 之间先进行充分的语义交互和增强，避免被时间步信号干扰。这相当于给文本编码器的输出做了一个轻量级的"后处理增强"，弥补了外部文本编码器（如 Qwen3.5-4B）可能存在的特征不足问题。

---

## Attention 组件创新点

### 创新点4：分离式 Q/K/V 投影替代合并式 QKV 投影

#### FLUX2 的做法
FLUX2 使用合并的 QKV 投影（一个 Linear 产生 Q/K/V）：

```python
# flux2/model.py SelfAttention 第384行
self.qkv = nn.Linear(dim, dim * 3, bias=False)
```

#### Z-Image 的做法
Z-Image 使用分离的 Q/K/V 投影（三个独立的 Linear），且 K/V 的维度可以与 Q 不同：

```python
# Z-Image/transformer.py ZImageAttention 第95-98行
self.to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
self.to_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
self.to_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
self.to_out = nn.ModuleList([nn.Linear(n_heads * self.head_dim, dim, bias=False)])
```

#### 区别与优点分析
- **功能性优化**：分离的 Q/K/V 投影是支持 GQA（Grouped Query Attention）的前提条件。当 `n_kv_heads < n_heads` 时，K 和 V 的投影维度比 Q 小，这在合并式 QKV 投影中无法实现。虽然默认配置 `n_kv_heads=30` 等于 `n_heads=30`（即标准 MHA），但分离设计保留了未来使用 GQA 的灵活性。

---

### 创新点5：支持 GQA（Grouped Query Attention）

#### FLUX2 的做法
FLUX2 仅支持标准 MHA（Multi-Head Attention），`num_heads` 同时用于 Q/K/V：

```python
# flux2/model.py SelfAttention.__init__() 第381-382行
self.num_heads = num_heads
self.qkv = nn.Linear(dim, dim * 3, bias=False)  # Q/K/V 维度相同
```

#### Z-Image 的做法
Z-Image 通过分离的 `n_heads` 和 `n_kv_heads` 参数原生支持 GQA：

```python
# Z-Image/transformer.py ZImageAttention.__init__() 第89-97行
def __init__(self, dim: int, n_heads: int, n_kv_heads: int, ...):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = dim // n_heads
    self.to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
    self.to_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)  # 可以比 Q 小
    self.to_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)  # 可以比 Q 小
```

默认配置中 `n_kv_heads=30 = n_heads=30`（标准 MHA），但架构已完全预留 GQA 支持。

#### 区别与优点分析
- **推理效率优化**：GQA 可以在保持相近生成质量的同时大幅减少 KV cache 的显存占用和计算量。当 `n_kv_heads < n_heads` 时（如设置 `n_kv_heads=6`，即 5 组 GQA），K/V 的参数量和计算量可以减少到 1/5。虽然当前默认配置未启用 GQA，但架构已做好准备，未来可以通过简单修改 `n_kv_heads` 参数来启用，无需修改任何代码。

---

## FFN 组件创新点

### 创新点6：SwiGLU FFN（三矩阵门控）替代双矩阵 SiLU Gated MLP

#### FLUX2 的做法
FLUX2 使用**双矩阵 SiLU Gated MLP**：

- **DoubleStreamBlock** 中：一个 Linear 投影到 `2 * mlp_hidden_dim`，然后 chunk 成两半做 SiLU gating：
  ```python
  # flux2/model.py DoubleStreamBlock 第546-550行
  self.img_mlp = nn.Sequential(
      nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False),  # 一个 Linear 产出 2*hidden_dim
      SiLUActivation(),  # chunk + SiLU gate: SiLU(x1) * x2
      nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
  )
  ```

- **SingleStreamBlock** 中：QKV 和 MLP 共享一个 `linear1`，MLP 部分同样使用 SiLU gating：
  ```python
  # flux2/model.py SingleStreamBlock 第453-459行
  self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim * 2, bias=False)
  # mlp 部分: SiLUActivation (chunk + SiLU gate)
  ```

#### Z-Image 的做法
Z-Image 使用**三矩阵 SwiGLU FFN**（与 LLaMA/Qwen 等主流 LLM 完全一致的设计）：

```python
# Z-Image/transformer.py FeedForward 第68-75行
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # gate 分支
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)    # output 投影
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)    # value 分支

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))  # SiLU(W1·x) * W3·x → W2
```

FFN hidden_dim 的计算方式：
```python
# Z-Image/transformer.py 第161行
self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
# hidden_dim = int(3840 / 3 * 8) = 10240
```

#### 区别与优点分析
- **生成效果优化**：三矩阵 SwiGLU（与 LLaMA/Qwen 一致的设计）相比双矩阵 SiLU Gated MLP，gate 分支（W1）和 value 分支（W3）使用**独立的权重矩阵**，提供了更强的表达能力。FLUX2 的做法中 gate 和 value 共享同一个输入投影矩阵（通过 chunk 分割），两者本质上是同一个线性变换的不同输出维度的划分，表达能力受限。三矩阵 SwiGLU 已在大量 LLM 研究中被验证优于传统的 FFN 和双矩阵 gated FFN。
- Z-Image 的 FFN 扩展比率约为 `8/3 ≈ 2.67`（hidden_dim=10240 vs dim=3840），但由于三矩阵结构（w1, w2, w3 三个矩阵），总参数量 = `3840×10240 + 10240×3840 + 3840×10240 ≈ 118M`，与 FLUX2 双矩阵结构的参数量相近。

---

## 归一化策略创新点

### 创新点7：Pre-RMSNorm + Post-RMSNorm 双重归一化替代 AdaLN Pre-Norm 单归一化

#### FLUX2 的做法
FLUX2 在 Attention 和 MLP 之前使用 **LayerNorm（`elementwise_affine=False`，无可学习参数）+ AdaLN 调制**，Attention/MLP 输出后直接通过 gate 做残差连接，没有 Post-Norm：

```python
# flux2/model.py DoubleStreamBlock
# Pre-Norm for Attention:
self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
# 应用: img_modulated = (1 + img_mod1_scale) * self.img_norm1(img) + img_mod1_shift

# 残差连接 (无 Post-Norm):
img = img + img_mod1_gate * self.img_attn.proj(img_attn)
```

#### Z-Image 的做法
Z-Image 使用 **Pre-RMSNorm + Post-RMSNorm** 的双重归一化结构，每个 block 有 4 个 RMSNorm：

```python
# Z-Image/transformer.py ZImageTransformerBlock 第163-166行
self.attention_norm1 = RMSNorm(dim, eps=norm_eps)  # Pre-Norm for Attention
self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)        # Pre-Norm for FFN
self.attention_norm2 = RMSNorm(dim, eps=norm_eps)   # Post-Norm for Attention
self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)         # Post-Norm for FFN

# forward() 第186-192行 (modulation=True 分支)
# Attention 路径：
attn_out = self.attention(
    self.attention_norm1(x) * scale_msa,       # Pre-RMSNorm → scale 调制
    attention_mask=attn_mask, freqs_cis=freqs_cis,
)
x = x + gate_msa * self.attention_norm2(attn_out)  # Post-RMSNorm → gate → 残差

# MLP 路径：
x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
#   Pre-RMSNorm → scale → FFN → Post-RMSNorm → gate → 残差
```

Z-Image 的 RMSNorm 带有**可学习的 scale 参数**：
```python
# Z-Image/transformer.py RMSNorm 第56-64行
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的 scale 参数

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight
```

#### 区别与优点分析
- **训练稳定性优化**：Post-RMSNorm 在残差连接之前对子层输出进行归一化，可以有效抑制深层网络中的梯度波动和特征尺度膨胀，提高训练稳定性。FLUX2 仅有 Pre-Norm，输出直接通过 gate 乘法加到残差路径上，如果 gate 值较大可能导致特征尺度不可控。
- **生成效果优化**：Z-Image 的 RMSNorm **带有可学习的 weight 参数**（`nn.Parameter(torch.ones(dim))`），可以自动学习每个维度的归一化权重；而 FLUX2 的 LayerNorm 使用 `elementwise_affine=False`（无可学习参数），完全依赖 AdaLN 的 shift/scale 来提供维度特定的调制。RMSNorm 相比 LayerNorm 计算更高效（不需要计算均值的减法操作）。
- Post-RMSNorm 的可学习 scale 参数可以部分弥补 Z-Image 去掉 shift 参数带来的表达能力损失（见创新点8）。

---

## AdaLN 调制创新点

### 创新点8：AdaLN 调制参数精简为 4 个，去掉 shift

#### FLUX2 的做法
FLUX2 DoubleStreamBlock 使用 **6 个调制参数**（shift1, scale1, gate1, shift2, scale2, gate2），SingleStreamBlock 使用 **3 个**（shift, scale, gate）：

```python
# flux2/model.py Modulation 第401-404行
self.is_double = double
self.multiplier = 6 if double else 3  # 双流块用 6 个，单流块用 3 个

# DoubleStreamBlock._prepare_qkv() 第580行
img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift  # 有 shift
```

#### Z-Image 的做法
Z-Image 仅使用 **4 个调制参数**（scale_msa, gate_msa, scale_mlp, gate_mlp），**没有 shift**：

```python
# Z-Image/transformer.py ZImageTransformerBlock 第169行
self.adaLN_modulation = nn.ModuleList([nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)])

# forward() 第180-184行
scale_msa, gate_msa, scale_mlp, gate_mlp = (
    self.adaLN_modulation[0](adaln_input).unsqueeze(1).chunk(4, dim=2)
)
gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

# 应用（无 shift）：
attn_out = self.attention(self.attention_norm1(x) * scale_msa, ...)  # 仅 scale，无 shift
```

#### 区别与优点分析
- **参数效率优化**：去掉 shift 参数将每层调制参数从 6 个（FLUX2 双流块）减少到 4 个（减少 33%），在深度为 30 层的网络中节省了可观的参数量。
- **生成效果可能不受影响**：Z-Image 通过 Post-RMSNorm（带可学习 weight 参数）和 Pre-RMSNorm（带可学习 weight 参数）来弥补去掉 shift 带来的表达能力损失。4 个 RMSNorm 共 `4 × dim` 个可学习参数可以在一定程度上替代 shift 的功能，因为 RMSNorm 的可学习 weight 可以实现维度特定的缩放。

---

### 创新点9：Gate 使用 Tanh 激活替代线性（无约束）Gate

#### FLUX2 的做法
FLUX2 的 gate 值直接从 Modulation 输出，没有显式激活函数约束，值域为 (-∞, +∞)：

```python
# flux2/model.py Modulation.forward() 第407-412行
def forward(self, vec):
    out = self.lin(nn.functional.silu(vec))
    ...
    out = out.chunk(self.multiplier, dim=-1)
    return out[:3], out[3:] if self.is_double else None
    # gate 值直接从 chunk 获得，无额外激活
```

#### Z-Image 的做法
Z-Image 的 gate 使用 **Tanh 激活**，值域约束在 (-1, +1)：

```python
# Z-Image/transformer.py 第183行
gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
```

#### 区别与优点分析
- **训练稳定性优化**：Tanh 将 gate 值约束在 (-1, 1) 范围内，防止极端的 gate 值导致梯度爆炸或特征缩放不稳定。这在训练初期尤其重要，可以避免模型在某些层完全关闭（gate→0）或过度放大（gate→极大值）残差连接的信号。FLUX2 的无约束 gate 虽然表达能力更强，但需要更仔细的学习率调优和梯度裁剪来防止训练不稳定。
- **生成效果优化**：约束在 (-1, 1) 的 gate 可以实现"软开关"效果，允许负 gate 值实现特征的部分反转，但不会出现过大的缩放因子。这种设计在 DiT 论文中也有类似的讨论（DiT 使用 zero-initialized gate）。

---

### 创新点10：AdaLN Embedding 维度独立于 hidden_size（ADALN_EMBED_DIM=256）

#### FLUX2 的做法
FLUX2 的时间步 embedding 维度直接等于 `hidden_size`（6144/4096/3072）：

```python
# flux2/model.py Flux2.__init__() 第69行
self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)
# MLPEmbedder: Linear(256 → hidden_size) → SiLU → Linear(hidden_size → hidden_size)
# 输出 vec 维度 = hidden_size

# Modulation 输入也是 hidden_size 维度：
self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, disable_bias=True)
# Modulation.lin: Linear(hidden_size → 6*hidden_size)
```

#### Z-Image 的做法
Z-Image 的时间步 embedding 维度被限制为固定的 `ADALN_EMBED_DIM = 256`（远小于 `dim=3840`）：

```python
# config/model.py 第3行
ADALN_EMBED_DIM = 256

# Z-Image/transformer.py 第322行
self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
# TimestepEmbedder: Linear(256 → 1024) → SiLU → Linear(1024 → min(3840,256)=256)
# 输出 adaln_input 维度 = 256

# 每层 Modulation 的输入维度也是 256：
# 第169行
self.adaLN_modulation = nn.ModuleList([nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)])
# Linear(256 → 4*3840=15360)
```

#### 区别与优点分析
- **参数效率优化**：FLUX2 中 `MLPEmbedder(256→6144→6144)` 约 `256×6144 + 6144×6144 ≈ 39.3M` 参数；而 Z-Image 的 `TimestepEmbedder(256→1024→256)` 仅约 `256×1024 + 1024×256 ≈ 0.52M` 参数。时间步 embedding 的参数量减少了约 75 倍。
- **设计理念差异**：Z-Image 认为时间步信息本身是低维的（仅需 256 维即可充分表达），不需要与 hidden_size 等大的中间表示。通过每层独立的 `Linear(256→4*dim)` 将低维调制信号投影到高维调制参数，实现了参数的高效利用——将参数预算分配到"每层不同的投影"而非"全局共享的高维 embedding"。

---

### 创新点11：每层独立 Modulation 替代全局共享 Modulation

#### FLUX2 的做法
FLUX2 使用 **3 个全局共享的 Modulation** 模块，所有层共享相同的调制参数：

```python
# flux2/model.py Flux2.__init__() 第98-108行
self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, disable_bias=True)
self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True, disable_bias=True)
self.single_stream_modulation = Modulation(self.hidden_size, double=False, disable_bias=True)

# forward() 第132-134行
double_block_mod_img = self.double_stream_modulation_img(vec)  # 计算一次，所有双流块共用
double_block_mod_txt = self.double_stream_modulation_txt(vec)  # 计算一次，所有双流块共用
single_block_mod, _ = self.single_stream_modulation(vec)       # 计算一次，所有单流块共用
```

#### Z-Image 的做法
Z-Image 每个 `ZImageTransformerBlock` 有**自己独立的** Modulation：

```python
# Z-Image/transformer.py ZImageTransformerBlock 第168-169行
if modulation:
    self.adaLN_modulation = nn.ModuleList([nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)])
```

每层在 forward 中独立计算调制参数：
```python
# 第180-181行
scale_msa, gate_msa, scale_mlp, gate_mlp = (
    self.adaLN_modulation[0](adaln_input).unsqueeze(1).chunk(4, dim=2)
)
```

#### 区别与优点分析
- **生成效果优化**：每层独立的 Modulation 允许不同深度的层对时间步信息做不同的响应。浅层可能更关注低频结构信息（大的空间布局），深层可能更关注高频细节信息（纹理和边缘），不同层需要不同的调制行为来适配。FLUX2 的全局共享方案所有层使用完全相同的调制参数，限制了各层的调制灵活性。
- **参数量考虑**：由于 Z-Image 的 Modulation 输入维度仅为 256（而非 hidden_size=3840 或更大），每层独立 Modulation 的额外参数 = `256 × 4 × 3840 ≈ 3.93M` × (30层主干 + 2层noise_refiner) ≈ 126M。相比之下，FLUX2 的全局 Modulation 参数 = `6144 × (6+6+3) × 6144 ≈ 566M`（但仅 3 个模块）。两种方案的 Modulation 总参数量在同一量级。

---

## 位置编码创新点

### 创新点12：3D RoPE（t, h, w）替代 4D RoPE（t, h, w, l）

#### FLUX2 的做法
FLUX2 使用 **4D RoPE**，维度分配为 `axes_dim=[32, 32, 32, 32]`（对应 t, h, w, l 四个轴），head_dim = 32×4 = 128：

```python
# flux2/model.py Flux2Params 第18行
axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])

# 文本 tokens 的位置 ID 包含 4 个坐标 (t, h, w, l)：
# flux2/sampling.py prc_txt() 第96-102行
coords = {
    "t": torch.arange(1) if t_coord is None else t_coord,
    "h": torch.arange(1),  # dummy = 0
    "w": torch.arange(1),  # dummy = 0
    "l": torch.arange(_l),  # 文本序列位置
}
x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])

# 图像 tokens 的位置 ID：
# flux2/sampling.py prc_img() 第142-151行
x_coords = {
    "t": torch.arange(1) if t_coord is None else t_coord,
    "h": torch.arange(h),
    "w": torch.arange(w),
    "l": torch.arange(1),  # dummy = 0
}
```

#### Z-Image 的做法
Z-Image 使用 **3D RoPE**，维度分配为 `axes_dims=[32, 48, 48]`（对应 t, h, w 三个轴），head_dim = 32+48+48 = 128：

```python
# Z-Image/config/model.py 第7-8行
ROPE_AXES_DIMS = [32, 48, 48]
ROPE_AXES_LENS = [1536, 512, 512]
```

文本 tokens 的位置 ID 为 3D (t, h, w)：
```python
# Z-Image/transformer.py patchify_and_embed() 第391-396行
cap_padded_pos_ids = self.create_coordinate_grid(
    size=(cap_ori_len + cap_padding_len, 1, 1),  # 3D: (seq, 1, 1) — 仅 t 轴变化
    start=(1, 0, 0),                              # t 从 1 开始（区分于图像的 t）
    device=device,
).flatten(0, 2)
```

图像 tokens 的位置 ID：
```python
# 第429-431行
image_ori_pos_ids = self.create_coordinate_grid(
    size=(F_tokens, H_tokens, W_tokens),  # 3D: (F, H, W)
    start=(cap_ori_len + cap_padding_len + 1, 0, 0),  # t 从文本长度之后开始
    device=device,
).flatten(0, 2)
```

#### 区别与优点分析
- **生成效果优化**：Z-Image 去掉了 FLUX2 的第 4 维 `l`（序列位置），将省出的 RoPE 维度分配给 `h` 和 `w` 轴（各 48 维 vs FLUX2 的各 32 维），使得空间位置编码的表达力更强（各轴增加了 50% 的维度）。对于图像生成任务，空间位置信息（h, w）是最关键的，`l` 轴在 FLUX2 中仅用于文本 token 的序列位置编码，而 Z-Image 通过 `t` 轴的坐标偏移（文本 t=1~N，图像 t=N+1~...）来区分不同类型的 token，避免了独立 `l` 轴的额外开销。
- Z-Image 仍保留了 `t` 轴（32 维），用于在编辑场景中区分不同的参考图（类似于 FLUX2 使用 t 轴区分参考图的方式）。

---

### 创新点13：RoPE theta=256 替代 theta=2000

#### FLUX2 的做法
```python
# flux2/model.py Flux2Params 第19行
theta: int = 2000
```

#### Z-Image 的做法
```python
# Z-Image/config/model.py 第6行
ROPE_THETA = 256.0
```

#### 区别与优点分析
- **生成效果优化**：更小的 theta（256 vs 2000）意味着 RoPE 频率 `1/(theta^(2i/d))` 在给定维度上更大，旋转角度变化更快，模型对相邻位置之间差异的感知更敏感。对于图像生成任务中常见的分辨率范围（如 512×512 或 1024×1024，对应 latent 空间的 32×32 或 64×64），较小的 theta 可以提供更精细的空间位置区分，有助于生成更准确的局部细节和纹理。
- theta=256 配合 `ROPE_AXES_LENS=[1536, 512, 512]` 的最大位置长度，确保在支持的最大分辨率范围内位置编码不会出现周期性混叠。

---

### 创新点14：复数乘法实现 RoPE 替代矩阵旋转实现 RoPE

#### FLUX2 的做法
FLUX2 使用 **2×2 旋转矩阵** 实现 RoPE：

```python
# flux2/model.py rope() 第818-825行
def rope(pos, dim, theta):
    scale = torch.arange(0, dim, 2, ...) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)  # 形成 2x2 旋转矩阵
    return out.float()

# apply_rope() 第828-833行
def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    # 2x2 矩阵-向量乘法
```

#### Z-Image 的做法
Z-Image 使用 **复数乘法** 实现 RoPE（与 LLaMA/Meta 标准实现一致）：

```python
# Z-Image/transformer.py RopeEmbedder.precompute_freqs_cis() 第236-244行
@staticmethod
def precompute_freqs_cis(dim, end, theta):
    freqs_cis = []
    for i, (d, e) in enumerate(zip(dim, end)):
        freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
        timestep = torch.arange(e, dtype=torch.float64)
        freqs = torch.outer(timestep, freqs).float()
        freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
        # 极坐标表示的复数: e^{i*θ}
        freqs_cis.append(freqs_cis_i)
    return freqs_cis

# apply_rotary_emb() 第78-83行
def apply_rotary_emb(x_in, freqs_cis):
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)  # 复数乘法实现旋转
        return x_out.type_as(x_in)
```

#### 区别与优点分析
- **计算效率优化**：复数乘法 `x * freqs_cis` 本质上等价于 2×2 旋转矩阵乘法，但实现更简洁。复数乘法在底层 CUDA 实现中通常更高效，因为它只需 4 次浮点乘法和 2 次加法（(a+bi)(c+di) = (ac-bd) + (ad+bc)i），而显式 2×2 矩阵乘法需要显式构造矩阵再进行乘法。
- 两种实现在数学上完全等价，生成效果无差异。复数实现与 LLaMA 标准一致，代码更简洁且更容易与 LLM 生态系统集成。

---

## 输出层创新点

### 创新点15：FinalLayer 仅使用 scale（无 shift）调制

#### FLUX2 的做法
FLUX2 的 `LastLayer` 使用 **shift + scale** 两个调制参数：

```python
# flux2/model.py LastLayer 第422-434行
self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

def forward(self, x, vec):
    mod = self.adaLN_modulation(vec)
    shift, scale = mod.chunk(2, dim=-1)
    x = (1 + scale) * self.norm_final(x) + shift  # 有 shift
    x = self.linear(x)
```

#### Z-Image 的做法
Z-Image 的 `FinalLayer` 仅使用 **scale**，无 shift：

```python
# Z-Image/transformer.py FinalLayer 第205-219行
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)  # 无 shift
        x = self.linear(x)
```

#### 区别与优点分析
- **设计一致性**：与 Z-Image 主干层中"无 shift"的 AdaLN 设计保持一致（创新点8），整个模型的调制策略统一为"仅 scale"。
- **参数效率优化**：FinalLayer 的 `adaLN_modulation` 输入维度为 `min(hidden_size, ADALN_EMBED_DIM) = 256`（而非 FLUX2 的 `hidden_size`），输出层使用 `bias=True`。FLUX2 的 `LastLayer.adaLN_modulation` 参数量 = `6144 × 2×6144 ≈ 75.5M`，而 Z-Image 的仅 `256 × 3840 ≈ 0.98M`，减少了约 77 倍。

---

## 序列处理创新点

### 创新点16：支持变长序列 Padding + Attention Mask

#### FLUX2 的做法
FLUX2 假设 batch 内所有样本具有相同的序列长度，不使用 attention mask：

```python
# flux2/model.py causal_attn_fn() 第806行 (num_ref_tokens=0 时)
attn_txt_img = F.scaled_dot_product_attention(q_txt_img, k_all, v_all, is_causal=False)
# 无 attn_mask 参数
```

#### Z-Image 的做法
Z-Image 原生支持 **batch 内变长序列**，通过 `pad_sequence` + `attention mask` 实现：

```python
# Z-Image/transformer.py forward() 第504-519行
x_item_seqlens = [len(_) for _ in x]
x = pad_sequence(x, batch_first=True, padding_value=0.0)  # 自动 padding 到 batch 最长

x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
for i, seq_len in enumerate(x_item_seqlens):
    x_attn_mask[i, :seq_len] = 1  # True = 有效 token

# 在 Attention 中传递 mask：
# ZImageAttention.forward() 第132-133行
hidden_states = dispatch_attention(
    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, ...
)
```

所有三个阶段（noise_refiner、context_refiner、主干层）都使用各自的 attention mask。

#### 区别与优点分析
- **训练效率优化**：支持变长序列意味着同一 batch 内可以包含不同分辨率的图像，无需将所有图像裁剪/缩放到相同尺寸。这大幅增加了训练数据的利用率和多样性，避免了因分辨率对齐导致的信息损失或不必要的计算浪费。
- **功能性优化**：在实际部署中，不同用户可能请求不同尺寸的图像，变长序列支持使得 batched inference 更加灵活高效，无需按分辨率分桶。

---

### 创新点17：Learnable Pad Token（可学习的填充令牌）

#### FLUX2 的做法
FLUX2 不使用 padding tokens（因为不支持变长序列）。

#### Z-Image 的做法
Z-Image 引入了**可学习的填充令牌**，分别用于图像和文本的 padding 位置：

```python
# Z-Image/transformer.py ZImageTransformer2DModel.__init__() 第328-329行
self.x_pad_token = nn.Parameter(torch.empty((1, dim)))     # 图像 padding token (dim=3840)
self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))   # 文本 padding token (dim=3840)

# forward() 第508行
x[torch.cat(x_inner_pad_mask)] = self.x_pad_token  # 将 padding 位置替换为可学习 token

# 第530行
cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
```

#### 区别与优点分析
- **生成效果优化**：可学习的 pad token 替代简单的零向量填充，使模型能够学习区分"真实 token"和"填充 token"的最优表示。零向量 padding 在 attention 中即使有 mask 也可能通过 softmax 的分母引入微小的干扰，而可学习 pad token 允许模型自动学习一个"中性"的 padding 表示。
- 这是 Z-Image 支持变长序列设计的配套组件（配合创新点16）。

---

## 输入/输出适配创新点

### 创新点18：多 Patch Size 支持（动态分辨率适配）

#### FLUX2 的做法
FLUX2 使用单一的 `img_in` 线性投影，不使用显式的 patch size 参数。Latent tokens 直接由 VAE 的空间维度展平得到（128 通道，每个空间位置一个 token）：

```python
# flux2/model.py Flux2.__init__() 第68行
self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
# in_channels=128, 每个 latent 空间位置 → 一个 token
```

#### Z-Image 的做法
Z-Image 支持多种 patch size 配置，并为每种 patch size 维护**独立的输入投影和输出层**：

```python
# Z-Image/transformer.py ZImageTransformer2DModel.__init__() 第269-306行
all_patch_size=(2,),      # 可配置多种 patch size
all_f_patch_size=(1,),    # 可配置多种帧 patch size

all_x_embedder = {}
all_final_layer = {}
for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
    x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
    all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder
    final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
    all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer
self.all_x_embedder = nn.ModuleDict(all_x_embedder)
self.all_final_layer = nn.ModuleDict(all_final_layer)
```

默认配置 `patch_size=2, f_patch_size=1`：输入投影 = `Linear(1×2×2×16=64, 3840)`。

在 forward 中根据传入的 patch_size 动态选择：
```python
# forward() 第505行
x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

# 第567行
unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
```

#### 区别与优点分析
- **功能性优化**：多 patch size 支持使模型可以在不同分辨率下使用不同的 patch 策略。较大的 patch size 可用于低分辨率快速生成（序列更短、推理更快），较小的 patch size 可用于高分辨率精细生成。
- **扩展性优化（视频生成）**：`f_patch_size` 参数（帧 patch size，默认为 1）为视频生成保留了时间维度的 patchify 能力。当 `f_patch_size > 1` 时，可以将多帧视频在时间维度折叠为更少的 tokens，这是图像模型扩展到视频生成的常见做法（如 CogVideoX）。

---

### 创新点19：使用传统 SD-style VAE（8x下采样, 16通道）替代 FLUX2 VAE（16x下采样, 128通道）

#### FLUX2 的做法
FLUX2 使用改进的 VAE，`z_channels=32`，经过 2×2 patch 折叠后 `in_channels=128`，有效下采样 16x，使用 BatchNorm 归一化：

```python
# flux2/autoencoder.py
z_channels: int = 32
# encode: 8x 卷积下采样 → patch(2×2) → 128ch, 总计 16x 下采样
# 使用 BatchNorm(affine=False) 归一化 latent
```

#### Z-Image 的做法
Z-Image 使用传统 SD-style VAE（`AutoencoderKL`），`latent_channels=4`（配置），但 Transformer 实际 `in_channels=16`（通过 patch_size=2 将 4ch latent 的 2×2 patch 折叠为 4×4=16 通道），有效下采样 8x（VAE）× 2x（patch）= 16x：

```python
# Z-Image/config/model.py
DEFAULT_VAE_SCALE_FACTOR = 8          # VAE 硬件下采样 8x
DEFAULT_VAE_LATENT_CHANNELS = 4       # VAE 输出 4 通道
DEFAULT_TRANSFORMER_IN_CHANNELS = 16  # Transformer 输入 16 通道 = 4 × 2 × 2 (patch)
DEFAULT_TRANSFORMER_PATCH_SIZE = (2,) # patch size = 2

# Z-Image/autoencoder.py AutoencoderKL
# 标准卷积式 Encoder-Decoder，scaling_factor=0.18215（SD 经典值）
# block_out_channels、layers_per_block 等参数与 SD 1.5/SDXL VAE 兼容
```

pipeline 中 latent 的生成逻辑：
```python
# Z-Image/pipeline.py 第185-192行
vae_scale = vae_scale_factor * 2  # 8 * 2 = 16
height_latent = 2 * (int(height) // vae_scale)  # 对于 1024: 2*(1024/16) = 128
width_latent = 2 * (int(width) // vae_scale)     # = 128
shape = (batch_size, transformer.in_channels, height_latent, width_latent)
# latent shape = [B, 16, 128, 128]
# 经 patch_size=2 后：tokens = (128/2) × (128/2) = 64 × 64 = 4096
# 每个 token 维度 = 16 × 2 × 2 = 64
```

#### 区别与优点分析
- **兼容性优化**：SD-style VAE 与现有大量预训练 VAE 权重（如 SD 1.5、SDXL 的 VAE）兼容，可以直接复用社区已有的高质量 VAE，降低了训练成本。不需要重新训练 FLUX2 风格的 VAE。
- **序列长度对比**：对于 1024×1024 图像，Z-Image 的 token 序列长度为 4096（64×64），每个 token 64 维（经 x_embedder 投影到 3840 维）；FLUX2 的序列长度也为 4096（64×64），每个 token 128 维（直接投影到 6144 维）。两者序列长度相同，但 Z-Image 每个 token 的原始信息量较低（64 vs 128），通过投影层来弥补。

---

## 条件注入创新点

### 创新点20：Timestep 先乘以 t_scale=1000 再送入 Embedder

#### FLUX2 的做法
FLUX2 的 `timestep_embedding` 函数内部有 `time_factor=1000.0` 的乘法：

```python
# flux2/model.py timestep_embedding() 第710-719行
def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    t = time_factor * t  # time_factor=1000.0，在 embedding 函数内部乘
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(...) / half)
    ...
```

#### Z-Image 的做法
Z-Image 在 forward 中**显式地先**将 timestep 乘以 `t_scale=1000`，然后送入 `TimestepEmbedder`：

```python
# Z-Image/transformer.py forward() 第487-488行
t = t * self.t_scale  # t_scale=1000.0，在 forward 中显式乘
t = self.t_embedder(t)

# TimestepEmbedder.timestep_embedding() 第35-45行
@staticmethod
def timestep_embedding(t, dim, max_period=MAX_PERIOD):  # MAX_PERIOD=10000
    # 直接使用 t 值（已经 ×1000），不再有 time_factor
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(...) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    ...
```

#### 区别与优点分析
- **功能等价**：两种做法在数学上完全等价（`t * 1000` 后进入 sin/cos embedding），区别仅在于乘法发生的位置。Z-Image 在模型 forward 中显式乘法使得代码意图更清晰，更易于理解和调试。

---

### 创新点21：文本条件投影使用 RMSNorm + Linear 替代单 Linear

#### FLUX2 的做法
FLUX2 使用单个 Linear 将文本 embedding 投影到 hidden_size：

```python
# flux2/model.py Flux2.__init__() 第70行
self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)
# context_in_dim=15360 → hidden_size=6144
```

#### Z-Image 的做法
Z-Image 使用 **RMSNorm + Linear** 的组合：

```python
# Z-Image/transformer.py ZImageTransformer2DModel.__init__() 第323-326行
self.cap_embedder = nn.Sequential(
    RMSNorm(cap_feat_dim, eps=norm_eps),  # cap_feat_dim=2560，先 RMSNorm 归一化
    nn.Linear(cap_feat_dim, dim, bias=True),  # 2560 → 3840，再 Linear 投影
)
```

#### 区别与优点分析
- **生成效果优化**：在 Linear 投影之前加入 RMSNorm 可以对文本编码器输出的特征进行尺度归一化，使得模型对文本编码器输出的数值范围变化更鲁棒。不同的文本编码器（Qwen3.5-4B vs Mistral-24B）或不同长度的文本输入可能产生不同尺度的特征，RMSNorm 可以将它们归一化到统一的尺度，有助于训练收敛。
- RMSNorm 的可学习 scale 参数还可以自动学习每个特征维度的重要性权重，相当于一个轻量级的特征选择机制。

---

## 参数策略创新点

### 创新点22：Linear 层部分使用 bias=True（区分条件路径与计算路径）

#### FLUX2 的做法
FLUX2 几乎所有 Linear 层都使用 `bias=False`：

```python
# flux2/model.py
self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)
# Modulation: nn.Linear(dim, multiplier*dim, bias=not disable_bias)  # disable_bias=True → bias=False
# SelfAttention: qkv = nn.Linear(dim, dim*3, bias=False), proj = nn.Linear(dim, dim, bias=False)
# MLP: nn.Linear(..., bias=False)
# LastLayer: nn.Linear(hidden_size, out_channels, bias=False), adaLN Linear bias=False
```

#### Z-Image 的做法
Z-Image 有选择地使用 `bias=True`，区分"条件注入路径"和"特征变换路径"：

| 组件 | bias | 路径类型 |
|------|------|---------|
| `TimestepEmbedder` 的 MLP（2个 Linear） | **True** | 条件注入 |
| `adaLN_modulation`（每层） | **True** | 条件注入 |
| `FinalLayer.linear` | **True** | 输出 |
| `FinalLayer.adaLN_modulation` | **True** | 条件注入 |
| `cap_embedder` 的 Linear | **True** | 条件注入 |
| `all_x_embedder` | **True** | 输入投影 |
| Attention 的 Q/K/V/Out | **False** | 特征变换 |
| FFN 的 w1/w2/w3 | **False** | 特征变换 |

```python
# 条件路径 (bias=True):
# TimestepEmbedder 第28-29行
nn.Linear(frequency_embedding_size, mid_size, bias=True)
nn.Linear(mid_size, out_size, bias=True)
# adaLN_modulation 第169行
nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)

# 特征变换路径 (bias=False):
# Attention 第95-98行
self.to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
# FFN 第70-72行
self.w1 = nn.Linear(dim, hidden_dim, bias=False)
```

#### 区别与优点分析
- **设计精细度优化**：Z-Image 区分了"条件注入路径"（bias=True）和"特征变换路径"（bias=False）。时间步嵌入、调制参数、输入投影等条件注入相关的层使用 bias=True，提供了额外的偏置自由度来更准确地表达条件信号；而 Attention 和 FFN 等核心计算路径使用 bias=False，减少参数并提高训练稳定性。这种混合策略比 FLUX2 的"全 False"更加精细。

---

## 工程优化创新点

### 创新点23：丰富的 Attention Backend 调度系统

#### FLUX2 的做法
FLUX2 直接使用 `F.scaled_dot_product_attention`（SDPA），没有后端调度机制：

```python
# flux2/model.py causal_attn_fn() 第786-806行
out = F.scaled_dot_product_attention(q_txt_img, k_all, v_all, is_causal=False)
```

#### Z-Image 的做法
Z-Image 实现了完整的 **Attention Backend 调度系统**，支持 8 种后端：

```python
# Z-Image/utils/attention.py AttentionBackend 第51-64行
class AttentionBackend(str, Enum):
    FLASH = "flash"                # Flash Attention 2
    FLASH_VARLEN = "flash_varlen"  # Flash Attention 2 变长序列
    FLASH_3 = "_flash_3"           # Flash Attention 3 (Hopper GPU)
    FLASH_VARLEN_3 = "_flash_varlen_3"  # Flash Attention 3 变长序列
    MPS_FLASH = "mps_flash"        # Apple Silicon MPS
    NATIVE = "native"              # PyTorch SDPA (默认)
    NATIVE_FLASH = "_native_flash" # SDPA Flash kernel
    NATIVE_MATH = "_native_math"   # SDPA Math kernel
```

通过 `dispatch_attention()` 函数统一调度，支持运行时动态切换：
```python
# 第508-514行
def set_attention_backend(backend):
    ZImageAttention._attention_backend = backend
```

特别值得注意的是，`FLASH_VARLEN` 和 `FLASH_VARLEN_3` 后端支持变长序列的高效处理：
```python
# 第222-273行
def _flash_varlen_attention(...):
    # 使用 flash_attn_varlen_func，通过 cu_seqlens 避免 padding 的计算浪费
```

#### 区别与优点分析
- **推理性能优化**：丰富的后端支持使得 Z-Image 可以在不同硬件上自动选择最优的 attention 实现。Flash Attention 3 在 Hopper GPU（H100/H200）上可以实现最高性能；Flash Attention 2 在 Ampere GPU（A100）上表现最好。
- **跨平台优化**：支持 Apple Silicon MPS 后端（通过 `mps-flash-attn` 包），使模型可以在 Mac M1/M2/M3/M4 上高效运行。
- **变长序列优化**：`flash_varlen` 后端专门针对变长序列场景优化（与创新点16配合），通过 `cu_seqlens` 机制避免了 padding 导致的计算浪费，在 batch 内序列长度差异较大时可以显著提升训练和推理效率。

---

## 推理策略创新点

### 创新点24：CFG Truncation 机制

#### FLUX2 的做法
FLUX2 使用标准 CFG 或 Guidance Embedding，全程使用相同的 guidance scale，没有 truncation 机制：

```python
# flux2/sampling.py denoise_cfg() 第404-405行
pred_uncond, pred_cond = pred.chunk(2)
pred = pred_uncond + guidance * (pred_cond - pred_uncond)  # 全程使用相同 guidance
```

#### Z-Image 的做法
Z-Image 支持 **CFG Truncation**，在去噪后期（低噪声阶段）自动关闭 CFG：

```python
# Z-Image/pipeline.py generate() 第227-231行
current_guidance_scale = guidance_scale
if do_classifier_free_guidance and cfg_truncation is not None and float(cfg_truncation) <= 1:
    if t_norm > cfg_truncation:
        current_guidance_scale = 0.0  # 在 t_norm > truncation 时关闭 CFG

apply_cfg = do_classifier_free_guidance and current_guidance_scale > 0
```

默认 `DEFAULT_CFG_TRUNCATION = 1.0`（即不 truncate，全程使用 CFG）。

#### 区别与优点分析
- **生成效果优化**：CFG 在去噪后期（高 SNR 阶段，`t_norm` 接近 1）容易引入过饱和和高频伪影。CFG Truncation 在去噪后期自动关闭 CFG，可以减少这些伪影，生成更自然的图像。这一技术已在 SD3/FLUX1 社区中被广泛验证有效。用户可以通过调整 `cfg_truncation` 参数（如设为 0.5）在图文一致性和图像自然度之间取得平衡。

---

### 创新点25：CFG Normalization 机制

#### FLUX2 的做法
FLUX2 不支持 CFG normalization。

#### Z-Image 的做法
Z-Image 支持 **CFG Normalization**（也称为 Rescaled CFG），对 CFG 输出的范数进行约束：

```python
# Z-Image/pipeline.py generate() 第263-269行
if cfg_normalization and float(cfg_normalization) > 0.0:
    ori_pos_norm = torch.linalg.vector_norm(pos)
    new_pos_norm = torch.linalg.vector_norm(pred)
    max_new_norm = ori_pos_norm * float(cfg_normalization)
    if new_pos_norm > max_new_norm:
        pred = pred * (max_new_norm / new_pos_norm)
```

#### 区别与优点分析
- **生成效果优化**：高 guidance_scale 会导致 CFG 后预测向量的范数过大，引起颜色过饱和和伪影。CFG Normalization 将 CFG 后的预测向量范数约束在原始条件预测范数的一定倍数内（`cfg_normalization` 参数），有效抑制过饱和问题，允许使用更高的 guidance_scale 来增强图文一致性而不牺牲图像质量。这一技术源自 "Common Diffusion Noise Schedules and Sample Steps are Flawed" 论文。

---

## 总结表

| # | 创新点 | FLUX2 做法 | Z-Image 做法 | 优化维度 |
|---|--------|-----------|-------------|---------|
| **整体架构** | | | | |
| 1 | 整体架构 | 双流8层+单流48层 | **Noise/Context Refiner 2层 + 纯单流30层** | 生成效果 + 计算效率 |
| 2 | 噪声预处理 | 无 | **Noise Refiner (2层, 带AdaLN调制)** | 生成效果 |
| 3 | 文本预处理 | 无 | **Context Refiner (2层, 无AdaLN调制)** | 生成效果 |
| **Attention 组件** | | | | |
| 4 | QKV 投影 | 合并 QKV (一个 Linear) | **分离 Q/K/V (三个 Linear)** | 功能性 (支持GQA) |
| 5 | GQA 支持 | 不支持 (仅 MHA) | **支持 (n_kv_heads 可配置)** | 推理效率 |
| **FFN 组件** | | | | |
| 6 | FFN 类型 | SiLU Gated (双矩阵, chunk 分割) | **SwiGLU (三矩阵, LLaMA式, 独立 gate/value)** | 生成效果 |
| **归一化策略** | | | | |
| 7 | 归一化结构 | Pre-LayerNorm (无可学习参数) | **Pre-RMSNorm + Post-RMSNorm (带可学习 weight)** | 训练稳定性 + 生成效果 |
| **AdaLN 调制** | | | | |
| 8 | 调制参数 | 6个 (shift,scale,gate ×2) | **4个 (scale,gate ×2), 无shift** | 参数效率 |
| 9 | Gate 激活 | 无约束 (线性) | **Tanh 约束到 (-1,1)** | 训练稳定性 |
| 10 | AdaLN 嵌入维度 | = hidden_size (6144) | **独立 ADALN_EMBED_DIM=256** | 参数效率 |
| 11 | Modulation 共享 | 全局共享 3 个 Modulation | **每层独立 Modulation** | 生成效果 |
| **位置编码** | | | | |
| 12 | RoPE 维度 | 4D (t,h,w,l), axes=[32,32,32,32] | **3D (t,h,w), axes=[32,48,48]** | 生成效果 (空间精度提升) |
| 13 | RoPE theta | 2000 | **256** | 生成效果 (位置敏感度提升) |
| 14 | RoPE 实现 | 2×2 矩阵旋转 | **复数乘法 (LLaMA式)** | 计算效率 |
| **输出层** | | | | |
| 15 | FinalLayer 调制 | shift + scale (2个参数) | **仅 scale (1个参数)** | 参数效率 + 设计一致性 |
| **序列处理** | | | | |
| 16 | 变长序列 | 不支持 (固定长度) | **支持 (padding + attention mask)** | 训练效率 + 功能性 |
| 17 | Pad Token | 无 | **可学习 pad token (图像+文本各一个)** | 生成效果 |
| **输入/输出适配** | | | | |
| 18 | 多 Patch Size | 不支持 (单一 img_in) | **支持 (ModuleDict, 含 f_patch_size 视频扩展)** | 功能性 + 扩展性 |
| 19 | VAE 类型 | FLUX2 VAE (16x, 128ch, BatchNorm) | **SD-style VAE (8x, 16ch→patch→64d, scaling_factor)** | 兼容性 |
| **条件注入** | | | | |
| 20 | Timestep 缩放 | embedding 内部 ×1000 | **forward 外部 ×1000** | 功能等价 (代码清晰度) |
| 21 | 文本投影 | 单 Linear (bias=False) | **RMSNorm + Linear (bias=True)** | 生成效果 (特征归一化鲁棒性) |
| **参数策略** | | | | |
| 22 | bias 策略 | 全部 bias=False | **条件路径 True, 计算路径 False** | 设计精细度 |
| **工程优化** | | | | |
| 23 | Attention 后端 | 仅 SDPA | **8种后端, 运行时调度 (含 FA2/3, varlen, MPS)** | 推理性能 + 跨平台 |
| **推理策略** | | | | |
| 24 | CFG Truncation | 不支持 | **支持 (去噪后期自动关闭 CFG)** | 生成效果 |
| 25 | CFG Normalization | 不支持 | **支持 (Rescaled CFG, 范数约束)** | 生成效果 |

---

## 总体设计哲学总结

Z-Image 的去噪模型在网络结构上相比 FLUX2 有以下主要设计理念差异：

1. **纯单流 + 前置 Refiner** 替代双流+单流的混合架构，通过"先独立精炼，再联合交互"的方式实现高效的多模态融合。Noise Refiner 带时间步调制、Context Refiner 不带时间步调制的不对称设计，精确反映了噪声信号和文本语义在去噪过程中的不同角色。

2. **LLM 风格的组件设计**：SwiGLU FFN（三矩阵门控）、分离 Q/K/V + GQA 支持、RMSNorm（带可学习参数）、复数乘法 RoPE 等，全面采用与现代 LLM（LLaMA/Qwen）一致的组件设计。这些设计已在大规模语言模型中被充分验证，移植到图像生成模型中可以获得类似的训练稳定性和表达能力优势。

3. **更精细的调制机制**：每层独立 Modulation（允许不同深度层对时间步做不同响应）、低维 AdaLN embedding（256维，参数效率更高）、Tanh gate 约束（防止训练不稳定）、无 shift 的精简调制参数（通过 Post-RMSNorm 的可学习 weight 弥补）。

4. **更强的训练/推理灵活性**：变长序列支持（配合可学习 pad token 和 attention mask）、多 patch size（含视频扩展 f_patch_size）、8 种 attention 后端运行时调度、CFG Truncation 和 CFG Normalization 等推理优化技术。

5. **兼容性优先的 VAE 选择**：使用 SD-style VAE（8x 下采样，16 通道）而非 FLUX2 的 128 通道 VAE，可以直接复用社区已有的大量预训练 VAE 权重，降低了从零训练的门槛。

---

> **分析依据的源代码文件**：
> - `Z-Image/src/zimage/transformer.py` — Z-Image 去噪模型核心定义（571行）
> - `Z-Image/src/config/model.py` — Z-Image 模型配置常量（45行）
> - `Z-Image/src/config/inference.py` — Z-Image 推理配置（8行）
> - `Z-Image/src/utils/attention.py` — Z-Image Attention 后端调度（516行）
> - `Z-Image/src/zimage/pipeline.py` — Z-Image 推理 Pipeline（293行）
> - `Z-Image/src/zimage/scheduler.py` — Z-Image Flow Match 调度器（150行）
> - `Z-Image/src/zimage/autoencoder.py` — Z-Image VAE（369行）
> - `flux2/src/flux2/model.py` — FLUX2 去噪模型定义（833行）
> - `flux2/src/flux2/autoencoder.py` — FLUX2 VAE
> - `flux2/src/flux2/sampling.py` — FLUX2 采样逻辑（442行）
