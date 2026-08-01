# Boogu-Image vs FLUX2 去噪模型架构对比分析

> 本报告聚焦于两个模型的 **去噪模型（Denoising Model / DIT Transformer）** 部分的网络结构差异，逐一列举 Boogu-Image 相对于 FLUX2 的所有变化创新点，并分析每个创新点的目的和效果。
>
> **代码依据**：
> - Boogu-Image: `/opt/nas/p/zhugechaoran/download/code/Boogu-Image/boogu/models/transformers/transformer_boogu.py` 及相关文件
> - FLUX2: `/opt/nas/p/zhugechaoran/download/code/flux2/src/flux2/model.py` 及相关文件
>
> 所有结论均经过代码复核校验。

---

## 目录

1. [整体架构层面的变化创新点](#一整体架构层面的变化创新点)
2. [单独组件层面的变化创新点](#二单独组件层面的变化创新点)
3. [创新点总结表](#三创新点总结表)

---

## 一、整体架构层面的变化创新点

### 创新点 1：引入三种 Refiner 子网络（Context Refiner / Noise Refiner / Reference Image Refiner）

**Boogu-Image 的做法**：

在主体双流/单流 Transformer 层之前，设置了三种独立的 Refiner 子网络，分别对不同模态的输入进行预处理精炼：

1. **Context Refiner**（`context_refiner`）：对指令/文本 embedding 进行自注意力精炼，`modulation=False`（不依赖时间步）。
2. **Noise Refiner**（`noise_refiner`）：对噪声图像的 patch embedding 进行精炼，`modulation=True`（依赖时间步）。
3. **Reference Image Refiner**（`ref_image_refiner`）：对参考图像的 patch embedding 进行精炼，`modulation=True`（依赖时间步）。参考图像被展平成独立 batch 进行处理。

每种 Refiner 默认 2 层（`num_refiner_layers=2`），且三种 Refiner 的权重完全独立。

**代码依据**（`transformer_boogu.py` 第 879-922 行）：
```python
self.noise_refiner = nn.ModuleList([
    BooguImageNoiseRefinerTransformerBlock(hidden_size, ..., modulation=True)
    for _ in range(num_refiner_layers)
])
self.ref_image_refiner = nn.ModuleList([
    BooguImageRefImgRefinerTransformerBlock(hidden_size, ..., modulation=True)
    for _ in range(num_refiner_layers)
])
self.context_refiner = nn.ModuleList([
    BooguImageContextRefinerTransformerBlock(hidden_size, ..., modulation=False)
    for _ in range(num_refiner_layers)
])
```

前向传播中（第 1363-1381 行）：
```python
# Context refinement
for layer in self.context_refiner:
    instruction_hidden_states = layer(instruction_hidden_states, ...)

# Image patch embedding and refinement (内含 noise_refiner 和 ref_image_refiner)
combined_img_hidden_states = self.img_patch_embed_and_refine(...)
```

在 `img_patch_embed_and_refine` 方法中（第 1043-1089 行），noise_refiner 处理噪声图像 patches，ref_image_refiner 将参考图像展平为独立 batch 进行处理。

**FLUX2 的做法**：

FLUX2 **没有任何 Refiner 子网络**。文本 embedding、噪声 latent、参考图像 latent 直接通过 `txt_in`/`img_in` 线性投影后进入双流块。

**代码依据**（`model.py` 第 136-141 行）：
```python
img = self.img_in(x)
txt = self.txt_in(ctx)
# 直接进入 double_blocks，无 refiner
for block in self.double_blocks:
    img, txt, _ = block.forward_kv_extract(img, txt, ...)
```

**优点分析**：

- **生成效果优化**：Refiner 在主体 Transformer 处理前对不同模态输入分别精炼，使得进入双流/单流层的特征质量更高。Context Refiner 不使用时间步调制，避免了文本特征受去噪阶段影响；Noise Refiner 和 Ref Image Refiner 使用时间步调制，使噪声/参考图像特征能感知当前去噪状态。参考图像被独立 batch 处理，避免不同参考图像之间的信息干扰。这些设计有助于提升生成和编辑的质量。

---

### 创新点 2：双流块（Double-Stream Block）中额外增加了 Image Self-Attention

**Boogu-Image 的做法**：

每个 `BooguImageDoubleStreamTransformerBlock` 包含 **两种注意力操作**：
1. `img_instruct_attn`（Joint Attention）：指令 tokens 和图像 tokens 的联合注意力（跨模态交互）
2. `img_self_attn`（Image Self-Attention）：图像 tokens 的独立自注意力

在前向传播中，先进行 Joint Attention 跨模态交互，然后图像流再进行一次独立的 Self-Attention。

**代码依据**（`transformer_boogu.py` 第 453-477 行）：
```python
self.img_instruct_attn = Attention(...)  # Joint Attention
self.img_self_attn = Attention(...)      # Image Self-Attention
```

前向传播（第 635-676 行）：
```python
# Step 2: joint attention on [instruct + img]
joint_attn_out = self.img_instruct_attn.processor(
    attn=self.img_instruct_attn,
    img_hidden_states=img_norm1_out,
    instruct_hidden_states=instruct_norm1_out, ...)
# Step 3: image self-attention
img_self_attn_out = self.img_self_attn(
    hidden_states=img_norm3_out, encoder_hidden_states=img_norm3_out, ...)
# Step 4: residual updates
img_hidden_states = img_hidden_states + gate_msa.tanh() * self.img_attn_norm(img_attn_out)
img_hidden_states = img_hidden_states + gate_self.tanh() * self.img_self_attn_norm(img_self_attn_out)
```

**FLUX2 的做法**：

FLUX2 的 `DoubleStreamBlock` **只有一次联合注意力**（Joint Attention），图像和文本的 QKV 拼接在一起计算注意力后分离，不存在额外的 Image Self-Attention。

**代码依据**（`model.py` 第 569-612 行）：
```python
q = torch.cat((txt_q, img_q), dim=2)
k = torch.cat((txt_k, img_k), dim=2)
v = torch.cat((txt_v, img_v), dim=2)
# 只有一次联合注意力，无额外 self-attention
```

**优点分析**：

- **生成效果优化**：额外的 Image Self-Attention 允许图像 tokens 在不受文本 tokens 干扰的情况下进行内部空间特征交互。这对于保持图像的空间一致性（如纹理连贯性、结构完整性）尤为重要。在图像编辑场景下，参考图像和噪声图像之间的独立自注意力可以加强图像级别的对齐。

---

### 创新点 3：双流块中使用完全独立的 QKV 投影和独立的输出投影

**Boogu-Image 的做法**：

在双流块的 Joint Attention 中，图像流和指令流使用 **完全独立的 Q/K/V 投影层** 和 **独立的输出投影层**：

- 图像流：`img_to_q`, `img_to_k`, `img_to_v`, `img_out`
- 指令流：`instruct_to_q`, `instruct_to_k`, `instruct_to_v`, `instruct_out`
- 共享的最终输出投影：`attn.to_out`

注意力计算流程：各自投影 → 拼接 QKV → 联合注意力 → 拆分 → 各自输出投影（`img_out`/`instruct_out`） → 共享输出投影（`to_out`）。

**代码依据**（`attention_processor.py` 第 66-78 行）：
```python
# BooguImageDoubleStreamSelfAttnProcessorFlash2Varlen
self.img_to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
self.img_to_k = nn.Linear(query_dim, kv_dim, bias=qkv_bias)
self.img_to_v = nn.Linear(query_dim, kv_dim, bias=qkv_bias)
self.instruct_to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
self.instruct_to_k = nn.Linear(query_dim, kv_dim, bias=qkv_bias)
self.instruct_to_v = nn.Linear(query_dim, kv_dim, bias=qkv_bias)
self.instruct_out = nn.Linear(query_dim, query_dim, bias=qkv_bias)
self.img_out = nn.Linear(query_dim, query_dim, bias=qkv_bias)
```

注意力后的输出投影（第 484-498 行）：
```python
instruct_projected = self.instruct_out(instruct_hidden_states)
img_projected = self.img_out(img_hidden_states)
# 合并后再过共享输出
hidden_states = attn.to_out[0](hidden_states)
```

同时，原始 Attention 模块中的 `to_q`/`to_k`/`to_v` 被删除（第 534-543 行）：
```python
del self.img_instruct_attn.to_k
del self.img_instruct_attn.to_v
del self.img_instruct_attn.to_q
```

**FLUX2 的做法**：

FLUX2 的双流块中，图像和文本各有独立的 `SelfAttention` 模块，每个模块内部使用 **共享的 `qkv` 线性层**（一个 Linear 同时产生 Q/K/V）和 **共享的 `proj` 输出层**。

**代码依据**（`model.py` 第 375-387 行）：
```python
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        self.qkv = nn.Linear(dim, dim * 3, bias=False)  # 共享 QKV
        self.proj = nn.Linear(dim, dim, bias=False)      # 共享输出
```

**优点分析**：

- **生成效果优化**：完全独立的 QKV 投影使图像流和指令流能够学习各自最优的查询/键/值表示，提供更灵活的跨模态特征交互能力。独立的输出投影（`img_out`/`instruct_out`）使注意力输出可以针对不同模态进行专门的后处理变换。这种设计增加了模型的表达能力，虽然会增加参数量，但可以使跨模态注意力更加精细。

---

### 创新点 4：双流块中指令流的 FFN 使用 shift+scale 调制

**Boogu-Image 的做法**：

在双流块中，图像流和指令流的 FFN 输入都使用了 **shift + scale** 调制。具体地，每个流使用两组 `LuminaRMSNormZero` 调制，其中第二组（`img_norm2`/`instruct_norm2`）产生的输出包含 `shift` 参数，用于对 FFN 输入进行仿射变换。

**代码依据**（`transformer_boogu.py` 第 617-700 行）：
```python
# 图像流 FFN 调制
img_norm2_out, img_shift_mlp, _, _ = self.img_norm2(img_hidden_states, temb)
img_mlp_input = (1 + img_scale_mlp.unsqueeze(1)) * img_norm2_out + img_shift_mlp.unsqueeze(1)

# 指令流 FFN 调制
instruct_norm2_out, instruct_shift_mlp, _, _ = self.instruct_norm2(instruct_hidden_states, temb)
instruct_mlp_input = (1 + instruct_scale_mlp.unsqueeze(1)) * instruct_norm2_out + instruct_shift_mlp.unsqueeze(1)
```

指令流有独立的 `instruct_norm1`（注意力调制）和 `instruct_norm2`（FFN 调制），共产生 8 个调制参数（scale_msa, gate_msa, scale_mlp, gate_mlp × 2组）。

**FLUX2 的做法**：

FLUX2 的双流块中，图像和文本分支的调制使用标准 AdaLN-Zero：`shift + scale + gate`，结构为 `(1 + scale) * Norm(x) + shift`。

**代码依据**（`model.py` 第 626-634 行）：
```python
img = img + img_mod1_gate * self.img_attn.proj(img_attn)
img = img + img_mod2_gate * self.img_mlp(
    (1 + img_mod2_scale) * (self.img_norm2(img)) + img_mod2_shift)
txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
txt = txt + txt_mod2_gate * self.txt_mlp(
    (1 + txt_mod2_scale) * (self.txt_norm2(txt)) + txt_mod2_shift)
```

**关键区别**：

两者都在 FFN 输入上使用了 shift+scale 调制，但 Boogu-Image 的双流块中，图像流额外有 **三组调制**（`img_norm1` 用于 Joint Attention，`img_norm2` 用于 FFN，`img_norm3` 用于 Image Self-Attention），而 FLUX2 只有两组（`mod1` 用于 Attention，`mod2` 用于 FFN）。这是因为 Boogu-Image 多了 Image Self-Attention，需要额外的调制参数。

**优点分析**：

- **生成效果优化**：更丰富的调制使得每个子模块（Joint Attention、Self-Attention、FFN）都可以根据时间步独立调整行为，增强了模型在不同去噪阶段的灵活性。

---

### 创新点 5：每层独立的调制（Per-Layer Modulation）vs 全局共享调制（Global Shared Modulation）

**Boogu-Image 的做法**：

每个 Transformer 块（无论是双流块还是单流块）都有 **自己独立的 `LuminaRMSNormZero` 调制层**。每个 `LuminaRMSNormZero` 包含一个线性投影 `nn.Linear(min(hidden_size, 1024), 4 * hidden_size)`，从时间步 embedding（`temb`）产生 4 个调制参数（`scale_msa, gate_msa, scale_mlp, gate_mlp`）。

**代码依据**（`block_lumina2.py` 第 39-71 行）：
```python
class LuminaRMSNormZero(nn.Module):
    def __init__(self, embedding_dim, norm_eps, norm_elementwise_affine):
        self.linear = nn.Linear(min(embedding_dim, 1024), 4 * embedding_dim, bias=True)
        self.norm = RMSNorm(embedding_dim, eps=norm_eps)
    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None])
        return x, gate_msa, scale_mlp, gate_mlp
```

每个单流块有 1 个 `LuminaRMSNormZero`（`norm1`），每个双流块有 5 个（`img_norm1`, `img_norm2`, `img_norm3`, `instruct_norm1`, `instruct_norm2`）。

**FLUX2 的做法**：

FLUX2 使用 **全局共享的 3 个 Modulation 层**，所有双流块共享 2 个（`double_stream_modulation_img`、`double_stream_modulation_txt`），所有单流块共享 1 个（`single_stream_modulation`）。

**代码依据**（`model.py` 第 98-108 行）：
```python
self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, disable_bias=True)
self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True, disable_bias=True)
self.single_stream_modulation = Modulation(self.hidden_size, double=False, disable_bias=True)
```

前向传播中一次性计算调制参数，所有块共享（第 132-134 行）：
```python
double_block_mod_img = self.double_stream_modulation_img(vec)
double_block_mod_txt = self.double_stream_modulation_txt(vec)
single_block_mod, _ = self.single_stream_modulation(vec)
```

**优点分析**：

- **生成效果优化**：每层独立的调制使不同深度的层能够根据时间步产生不同的调制行为，理论上表达能力更强。深层可能需要不同于浅层的时间步响应策略，独立调制使这成为可能。
- **代价**：参数量增加。但由于 `LuminaRMSNormZero` 的输入维度被限制为 `min(hidden_size, 1024)=1024`，每个调制层的参数量相对可控。

---

### 创新点 6：tanh Gate 机制（Lumina 风格）vs 直接乘法 Gate

**Boogu-Image 的做法**：

在残差连接中，gate 值通过 **`tanh` 激活** 后再与注意力/FFN 输出相乘：

**代码依据**（`transformer_boogu.py` 第 355-362 行，单流块）：
```python
hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
```

以及双流块中（第 671-684 行）：
```python
img_hidden_states = img_hidden_states + img_gate_msa.unsqueeze(1).tanh() * self.img_attn_norm(img_attn_out)
img_hidden_states = img_hidden_states + img_gate_self.unsqueeze(1).tanh() * self.img_self_attn_norm(img_self_attn_out)
img_hidden_states = img_hidden_states + img_gate_mlp.unsqueeze(1).tanh() * self.img_ffn_norm2(img_mlp_out)
```

**FLUX2 的做法**：

FLUX2 的 gate 值直接与输出相乘，**不经过 `tanh` 激活**：

**代码依据**（`model.py` 第 626-634 行）：
```python
img = img + img_mod1_gate * self.img_attn.proj(img_attn)
img = img + img_mod2_gate * self.img_mlp(...)
```

以及单流块中（第 482-484 行）：
```python
output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
return x + mod_gate * output
```

**优点分析**：

- **生成效果优化（训练稳定性）**：`tanh` 将 gate 值限制在 [-1, 1] 范围内，防止了 gate 值过大导致的梯度爆炸或训练不稳定。这提供了一种内在的梯度裁剪机制，尤其在训练初期（当 gate 线性层初始化为零时，`tanh(0)=0`，残差贡献为零）可以保证更平滑的训练开始。这是训练稳定性维度的优化，间接有助于更好的最终生成效果。

---

### 创新点 7：GQA（Grouped Query Attention）vs 标准 MHA

**Boogu-Image 的做法**：

使用 **Grouped Query Attention (GQA)**，Q 头数为 24，KV 头数为 8，比例为 3:1。

**代码依据**（`transformer_boogu.py` 第 813-814 行）：
```python
num_attention_heads: int = 24,
num_kv_heads: int = 8,
```

Attention 初始化（第 219-230 行）：
```python
self.attn = Attention(
    query_dim=dim,
    dim_head=dim // num_attention_heads,
    heads=num_attention_heads,
    kv_heads=num_kv_heads,  # GQA: kv_heads < heads
    ...)
```

在 attention processor 中通过 `repeat_interleave` 扩展 KV heads（`attention_processor.py` 第 1260-1261 行）：
```python
key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)
```

**FLUX2 的做法**：

使用 **标准 MHA（Multi-Head Attention）**，Q/K/V 头数相同，均为 `num_heads=48`。

**代码依据**（`model.py` 第 15 行）：
```python
num_heads: int = 48
```

SelfAttention（第 384 行）：
```python
self.qkv = nn.Linear(dim, dim * 3, bias=False)  # Q/K/V 同维度
```

**优点分析**：

- **计算效率优化**：GQA 减少了 KV 的计算和显存占用（KV 参数量减少为原来的 1/3），在长序列场景（高分辨率图像 + 多参考图像）下显著降低计算成本，同时研究表明 GQA 能在保持大部分表达能力的同时大幅提高推理效率。这主要是效率维度的优化，对生成效果的影响通常中性或轻微。

---

### 创新点 8：RMSNorm vs LayerNorm

**Boogu-Image 的做法**：

全面使用 **RMSNorm** 作为归一化层（支持 Triton 加速的融合版本）。

**代码依据**（`transformer_boogu.py` 第 56-58 行）：
```python
if is_triton_available() and ("cuda" in os.getenv("device", "cpu")):
    from ...ops.triton.layer_norm import RMSNorm
else:
    from torch.nn import RMSNorm
```

在各个组件中广泛使用（如 `transformer_boogu.py` 第 248-250 行）：
```python
self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
self.norm2 = RMSNorm(dim, eps=norm_eps)
self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)
```

**FLUX2 的做法**：

使用 **无参数 LayerNorm**（`elementwise_affine=False`）。

**代码依据**（`model.py` 第 537 行）：
```python
self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
```

注意：FLUX2 的 QKNorm 使用了带可学习 scale 参数的 RMSNorm（第 734-743 行）：
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim):
        self.scale = nn.Parameter(torch.ones(dim))
```

但主体 Transformer 块的归一化使用 LayerNorm。

**优点分析**：

- **计算效率优化**：RMSNorm 比 LayerNorm 计算更快（省去了均值计算和均值减法），且搭配 Triton 融合实现可以进一步加速。对生成效果通常影响中性，主要是效率优化。

---

### 创新点 9：SwiGLU FFN 的三线性层结构 vs SiLU Gated 两线性层结构

**Boogu-Image 的做法**：

FFN 使用 **SwiGLU（三个线性层）**：`linear_1` 和 `linear_3` 并行计算两个分支，SwiGLU 激活后通过 `linear_2` 输出。

**代码依据**（`block_lumina2.py` 第 125-174 行）：
```python
class LuminaFeedForward(nn.Module):
    def __init__(self, dim, inner_dim, multiple_of, ffn_dim_multiplier):
        self.linear_1 = nn.Linear(dim, inner_dim, bias=False)
        self.linear_2 = nn.Linear(inner_dim, dim, bias=False)
        self.linear_3 = nn.Linear(dim, inner_dim, bias=False)
    def forward(self, x):
        h1, h2 = self.linear_1(x), self.linear_3(x)
        return self.linear_2(swiglu(h1, h2))
```

其中 `swiglu` 实现（`components.py`）：
```python
def swiglu(x, y):
    return F.silu(x.float(), inplace=False).to(x.dtype) * y
```

默认 `inner_dim = 4 * dim`，且通过 `multiple_of=256` 对齐。支持 flash_attn 的融合 SwiGLU 加速。

**FLUX2 的做法**：

FFN 使用 **SiLU Gated Activation（两个线性层 + chunk）**：`linear1` 的输出维度为 `mlp_hidden_dim * 2`，chunk 成两半后做 SiLU gating。

**代码依据**（`model.py` 第 390-397 行，546-549 行）：
```python
class SiLUActivation(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2

self.img_mlp = nn.Sequential(
    nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False),
    SiLUActivation(),
    nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
)
```

**优点分析**：

- **生成效果优化（可能）**：两者数学上等价（都是 `SiLU(Wx) * Vx → Output`），但 Boogu-Image 的三线性层结构使 `linear_1` 和 `linear_3` 完全独立参数化，而 FLUX2 的两线性层结构中 gate 和 value 分支共享同一个输入投影矩阵的不同行。独立参数化理论上有更大的表达自由度。另外 Boogu-Image 支持 flash_attn 的融合 SwiGLU 内核，在效率上也有优势。

---

### 创新点 10：注意力输出后的 Post-Norm（RMSNorm）

**Boogu-Image 的做法**：

在注意力输出进入残差连接之前，先通过一个 **RMSNorm**（`norm2`/`img_attn_norm`/`img_self_attn_norm` 等）进行归一化。FFN 输出同样如此（`ffn_norm2`/`img_ffn_norm2`）。

**代码依据**（`transformer_boogu.py` 第 355-362 行）：
```python
# 单流块
hidden_states = hidden_states + gate_msa.tanh() * self.norm2(attn_output)
hidden_states = hidden_states + gate_mlp.tanh() * self.ffn_norm2(mlp_output)
```

双流块（第 671-684 行）：
```python
img_hidden_states = img_hidden_states + gate_msa.tanh() * self.img_attn_norm(img_attn_out)
img_hidden_states = img_hidden_states + gate_self.tanh() * self.img_self_attn_norm(img_self_attn_out)
img_hidden_states = img_hidden_states + gate_mlp.tanh() * self.img_ffn_norm2(img_mlp_out)
```

同时，FFN 输入也有一个 Pre-Norm（`ffn_norm1`/`img_ffn_norm1`）。

**FLUX2 的做法**：

FLUX2 **不对注意力输出和 FFN 输出** 进行额外的归一化，直接通过 gate 后加入残差：

**代码依据**（`model.py` 第 626-634 行）：
```python
img = img + img_mod1_gate * self.img_attn.proj(img_attn)  # 无额外 norm
img = img + img_mod2_gate * self.img_mlp(...)              # 无额外 norm
```

**优点分析**：

- **生成效果优化（训练稳定性）**：对注意力和 FFN 输出进行 Post-Norm 可以在残差累加前稳定特征的尺度，防止深层网络中特征幅值的不断增长，有助于训练稳定性，间接影响最终生成效果。

---

### 创新点 11：FFN 输入的 Pre-Norm 后再通过 scale 调制

**Boogu-Image 的做法**：

FFN 输入的处理流程为：`hidden_states → ffn_norm1（RMSNorm）→ ×(1 + scale_mlp) → FFN`。

**代码依据**（`transformer_boogu.py` 第 358-360 行）：
```python
mlp_output = self.feed_forward(
    self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1))
)
```

双流块中更复杂（第 678-681 行）：
```python
img_mlp_input = (1 + img_scale_mlp.unsqueeze(1)) * img_norm2_out + img_shift_mlp.unsqueeze(1)
img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_mlp_input))
```

**FLUX2 的做法**：

FFN 输入流程为：`img → LayerNorm → ×(1 + scale) + shift → FFN`，归一化和调制在同一步完成。

**优点分析**：

- Boogu-Image 在双流块中对 FFN 输入应用了**双层 Norm + 调制**（先经过 `norm2` 调制 + shift，再经过 `ffn_norm1`），比 FLUX2 的单层 Norm + 调制更精细，但区别不大。

---

### 创新点 12：3 轴复数形式 RoPE vs 4 轴旋转矩阵形式 RoPE

**Boogu-Image 的做法**：

使用 **3 轴复数形式 RoPE**：
- 3 轴维度分配：`axes_dim_rope = (40, 40, 40)`，对应 `(文本/时间, 行, 列)` 三个维度
- theta 基频：`10000`
- 使用复数形式应用 RoPE（Lumina 风格）

**代码依据**（`rope.py` 第 223-235 行，`transformer_boogu.py` 第 817 行）：
```python
axes_dim_rope: Tuple[int, int, int] = (40, 40, 40),  # 3轴，合计 120 = 2304/24
```

RoPE 应用方式（`embeddings.py` 第 127-133 行）：
```python
# 复数形式 (used for lumina)
x_rotated = torch.view_as_complex(x.float().reshape(..., x.shape[-1] // 2, 2))
freqs_cis = freqs_cis.unsqueeze(2)
x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
```

位置 ID 构建（`rope.py` 第 296-361 行）：
- 文本 tokens：3 个轴都使用相同的 1D 位置（`repeat("l -> l 3")`）
- 图像 tokens：第 0 轴用 `pe_shift`（文本/图像区分），第 1 轴用行坐标，第 2 轴用列坐标

**FLUX2 的做法**：

使用 **4 轴旋转矩阵形式 RoPE**：
- 4 轴维度分配：`axes_dim = [32, 32, 32, 32]`，对应 `(t, h, w, l)` 四个维度
- theta 基频：`2000`
- 使用 2×2 旋转矩阵形式应用 RoPE

**代码依据**（`model.py` 第 18 行，818-833 行）：
```python
axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
theta: int = 2000

def rope(pos, dim, theta):
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)  # 2x2 旋转矩阵

def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
```

位置 ID 构建（`sampling.py` 第 93-103 行，141-151 行）：
- 文本 tokens：`(t, h=0, w=0, l=位置索引)` — 4D
- 图像 tokens：`(t, h=行, w=列, l=0)` — 4D

**优点分析**：

- **生成效果优化（可能）**：3 轴 vs 4 轴各有优劣。FLUX2 的 4 轴设计将文本序列位置和图像空间位置分离到不同轴（文本用 `l` 轴，图像用 `h,w` 轴），而 `t` 轴用于区分不同参考图像的时间坐标。Boogu-Image 的 3 轴设计更紧凑，文本和图像共享 3 个轴（文本的 3 轴相同，图像用第 0 轴区分图像、第 1/2 轴编码空间位置）。3 轴每个轴有 40 维（vs 4 轴每轴 32 维），每个轴的表达能力更强。复数形式 RoPE 在数学上与旋转矩阵形式等价，但实现更简洁。theta=10000 vs theta=2000 会影响位置编码的频率范围，theta 更大意味着低频分量更丰富，可能对长序列的位置区分能力更好。总体上这是不同的设计选择，各有取舍。

---

### 创新点 13：独立的参考图像 Patch Embedder 和 Image Index Embedding

**Boogu-Image 的做法**：

噪声图像和参考图像使用 **独立的 Patch Embedding 层**：
- `x_embedder`：噪声图像的 patch 嵌入
- `ref_image_patch_embedder`：参考图像的 patch 嵌入

此外，引入 `image_index_embedding`（可学习参数，形状 `[5, hidden_size]`），为最多 5 张参考图像添加区分标识。

**代码依据**（`transformer_boogu.py` 第 861-970 行）：
```python
self.x_embedder = nn.Linear(
    in_features=patch_size * patch_size * in_channels, out_features=hidden_size)
self.ref_image_patch_embedder = nn.Linear(
    in_features=patch_size * patch_size * in_channels, out_features=hidden_size)
self.image_index_embedding = nn.Parameter(torch.randn(5, hidden_size))  # max 5 ref images
```

使用（第 1031-1041 行）：
```python
hidden_states = self.x_embedder(hidden_states)  # 噪声图像
ref_image_hidden_states = self.ref_image_patch_embedder(ref_image_hidden_states)  # 参考图像
# 为每张参考图像添加索引嵌入
for j, ref_img_len in enumerate(l_effective_ref_img_len[i]):
    ref_image_hidden_states[i, shift:shift+ref_img_len, :] += self.image_index_embedding[j]
```

**FLUX2 的做法**：

FLUX2 使用 **同一个 `img_in` 层** 处理噪声图像和参考图像的 latent tokens，通过 **不同的时间坐标**（`t_off = [10, 20, 30, ...]`）在 4D RoPE 中区分不同参考图像。

**代码依据**（`model.py` 第 68 行，`sampling.py` 第 76 行）：
```python
self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)  # 共享
# 参考图通过时间坐标区分
t_off = [scale + scale * t for t in torch.arange(0, len(encoded_refs))]
```

**优点分析**：

- **生成效果优化**：独立的 Patch Embedder 使噪声图像和参考图像学习不同的特征映射，更好地适应两者不同的分布特性（噪声 vs 干净图像）。Image Index Embedding 显式地区分不同参考图像的身份，比仅通过 RoPE 时间坐标的隐式区分更直接。这有助于多参考图编辑场景中更精确地利用不同参考图的信息。

---

### 创新点 14：参考图像的处理方式——原生序列融合 vs 因果注意力/KV Cache

**Boogu-Image 的做法**：

参考图像经过独立的 `ref_image_patch_embedder` 和 `ref_image_refiner` 处理后，与噪声图像的 patches **直接拼接** 形成 `combined_img_hidden_states`，然后统一进入双流块和单流块。所有 tokens（指令、参考图像、噪声图像）在注意力计算中 **完全可见**（全局注意力）。

**代码依据**（`transformer_boogu.py` 第 1102-1114 行）：
```python
combined_img_hidden_states = hidden_states.new_zeros(batch_size, max_combined_img_len, ...)
for i, (ref_img_len, img_len) in enumerate(zip(l_effective_ref_img_len, l_effective_img_len)):
    combined_img_hidden_states[i, :sum(ref_img_len)] = ref_image_hidden_states[i, :sum(ref_img_len)]
    combined_img_hidden_states[i, sum(ref_img_len):sum(ref_img_len)+img_len] = hidden_states[i, :img_len]
```

**FLUX2 的做法**：

FLUX2 使用 **因果注意力**（`causal_attn_fn`）处理参考图像：
- 参考图像 tokens 只能自注意力（不能看到噪声图像和文本）
- 噪声图像和文本 tokens 可以看到所有 tokens（包括参考图像）

同时支持 **KV Cache** 加速：第一步提取参考图像的 K/V 缓存，后续步骤复用。

**代码依据**（`model.py` 第 758-815 行）：
```python
def causal_attn_fn(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache=None):
    # txt+img attend to all keys
    q_txt_img = torch.cat([q_txt, q_img], dim=2)
    k_all = torch.cat([k_txt, k_ref, k_img], dim=2)
    attn_txt_img = F.scaled_dot_product_attention(q_txt_img, k_all, v_all)
    # ref only attends to itself
    attn_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)
```

**优点分析**：

- **生成效果优化（可能）**：Boogu-Image 的全局注意力方式让参考图像 tokens 能够直接参与所有注意力交互（包括与噪声图像和文本的双向交互），信息流动更加充分。在图像编辑场景中，参考图像能够"看到"当前噪声图像的状态并据此调整自身表示，理论上可以实现更精细的编辑控制。但代价是计算量更大（无法使用 KV Cache 加速）。FLUX2 的因果注意力则通过限制参考图像的可见范围来防止信息泄漏，并通过 KV Cache 加速推理。这是一个生成效果 vs 推理效率的权衡。

---

### 创新点 15：输出层——LuminaLayerNormContinuous vs AdaLN LastLayer

**Boogu-Image 的做法**：

使用 `LuminaLayerNormContinuous` 作为输出层：先将时间步 embedding 投影为 scale → 乘以归一化后的隐状态 → 线性投影到输出维度。

**代码依据**（`block_lumina2.py` 第 74-122 行）：
```python
class LuminaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim, conditioning_embedding_dim, ...):
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=bias)  # 投影到输出维度
    def forward(self, x, conditioning_embedding):
        scale = self.linear_1(self.silu(conditioning_embedding))
        x = self.norm(x) * (1 + scale)[:, None, :]
        x = self.linear_2(x)
        return x
```

初始化（`transformer_boogu.py` 第 958-965 行）：
```python
self.norm_out = LuminaLayerNormContinuous(
    embedding_dim=hidden_size,
    conditioning_embedding_dim=min(hidden_size, 1024),  # 输入维度限制为 1024
    elementwise_affine=False,
    out_dim=patch_size * patch_size * self.out_channels,
)
```

只使用 **scale**（无 shift），条件输入维度限制为 `min(hidden_size, 1024)`。

**FLUX2 的做法**：

使用 `LastLayer`：标准 AdaLN 输出层，包含 shift 和 scale 两个调制参数。

**代码依据**（`model.py` 第 415-434 行）：
```python
class LastLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))
    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = (1 + scale) * self.norm_final(x) + shift
        x = self.linear(x)
        return x
```

使用 **shift + scale** 两个调制参数，条件输入维度为完整的 `hidden_size`。

**优点分析**：

- 设计选择差异。Boogu-Image 的输出层只用 scale 不用 shift，且条件输入维度受限为 1024，更轻量。FLUX2 的输出层使用完整的 shift+scale，条件表达能力更强。这是一个参数量 vs 表达能力的权衡，影响较小。

---

### 创新点 16：时间步嵌入结构——Timesteps + TimestepEmbedding vs 正弦嵌入 + MLPEmbedder

**Boogu-Image 的做法**：

使用 `Lumina2CombinedTimestepCaptionEmbedding`，包含：
1. `Timesteps`（diffusers 标准，可翻转 sin/cos）→ `TimestepEmbedding`（两层 Linear + SiLU，带 bias）
2. `CaptionEmbedder`（RMSNorm → Linear，将指令特征投影到 hidden_size）

输出的 `temb` 维度为 `min(hidden_size, 1024)`。

**代码依据**（`block_lumina2.py` 第 177-219 行）：
```python
class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
    self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0.0, scale=timestep_scale)
    self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=min(hidden_size, 1024))
    self.caption_embedder = nn.Sequential(RMSNorm(instruction_feat_dim), nn.Linear(instruction_feat_dim, hidden_size, bias=True))
```

`TimestepEmbedding`（`embeddings.py`）：
```python
class TimestepEmbedding(nn.Module):
    self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)  # 有 bias
    self.act = get_activation("silu")
    self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)  # 有 bias
```

**FLUX2 的做法**：

使用 `timestep_embedding`（正弦嵌入函数）→ `MLPEmbedder`（两层 Linear + SiLU，无 bias）。时间步嵌入和 guidance 嵌入相加，产生 `vec`，维度为完整的 `hidden_size`。

**代码依据**（`model.py` 第 69-74 行，683-691 行，710-731 行）：
```python
self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)
self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)

class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim, disable_bias=False):
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=not disable_bias)  # 无 bias
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=not disable_bias)  # 无 bias
```

**优点分析**：

- **设计差异**：Boogu-Image 的时间步嵌入输出维度限制为 `min(hidden_size, 1024)=1024`（瓶颈设计），而 FLUX2 的时间步嵌入维度为完整的 `hidden_size=6144`。Boogu-Image 使用 bias，FLUX2 不使用 bias。Boogu-Image 将指令特征和时间步分开输出（`temb` 和 `caption_embed`），而 FLUX2 将时间步和 guidance 合并为单一的 `vec`。瓶颈设计减少了调制参数量但可能限制表达能力。

---

### 创新点 17：无 Guidance Embedding

**Boogu-Image 的做法**：

去噪模型内部 **没有 guidance embedding**。CFG（Classifier-Free Guidance）完全在推理管线层面通过多次前向传播和差值计算实现（Double Guidance 机制：text_guidance_scale + image_guidance_scale）。

**代码依据**：在 `transformer_boogu.py` 的 `BooguImageTransformer2DModel.__init__` 中，没有任何 `guidance_in` 或类似组件。

**FLUX2 的做法**：

FLUX2 的去噪模型内置了可选的 **Guidance Embedding**（`guidance_in`），将 guidance 值编码为嵌入向量并加到时间步嵌入上。

**代码依据**（`model.py` 第 72-74 行）：
```python
self.use_guidance_embed = params.use_guidance_embed
if self.use_guidance_embed:
    self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)
```

**优点分析**：

- **不同维度的优化**：Boogu-Image 不在模型内部注入 guidance，而是在推理管线层面实现更复杂的 Double Guidance（独立控制文本引导和图像引导强度）。这使得模型本身更简洁，同时引导策略更灵活（可以在推理时自由调整，不需要在训练时确定）。FLUX2 的 Guidance Embedding 是蒸馏训练的产物（将 CFG 的效果内化到模型中以减少推理时的前向传播次数），是推理效率维度的优化。

---

### 创新点 18：模型规模和超参数配置

**Boogu-Image 的配置**：

| 参数 | 值 |
|------|-----|
| `hidden_size` | 2304 |
| `num_layers` (总层数) | 26 |
| `num_double_stream_layers` | 2 |
| `num_single_stream_layers` | 24 |
| `num_refiner_layers` | 2（每种 Refiner） |
| `num_attention_heads` | 24 |
| `num_kv_heads` | 8 |
| `head_dim` | 96 (=2304/24) |
| `in_channels` | 16 |
| `patch_size` | 2 |

**FLUX2 [dev] 的配置**：

| 参数 | 值 |
|------|-----|
| `hidden_size` | 6144 |
| `depth` (双流层数) | 8 |
| `depth_single_blocks` (单流层数) | 48 |
| `num_heads` | 48 |
| `head_dim` | 128 (=6144/48) |
| `in_channels` | 128 |
| `mlp_ratio` | 3.0 |

**关键差异**：

1. **模型更轻量**：Boogu-Image 的 hidden_size (2304) 远小于 FLUX2 (6144)，总层数 (26+6 refiners=32) 也少于 FLUX2 (56)，整体参数量更小。
2. **更少的双流层**：Boogu-Image 只有 2 层双流块，FLUX2 有 8 层。Boogu-Image 通过独立的 Refiner 和更强的双流块设计（额外 self-attention）来弥补层数的减少。
3. **Latent 维度差异**：Boogu-Image 的 `in_channels=16`（使用 FLUX1 风格 VAE），FLUX2 的 `in_channels=128`（使用自定义 VAE with patch 操作）。

**优点分析**：

- **推理效率优化**：模型更轻量，推理速度更快，显存占用更小。
- **生成效果**：需要通过实际训练和评测来比较，更小的模型在同等数据和训练量下可能生成效果不及更大模型，但通过架构创新（Refiner、额外 Self-Attention、GQA 等）可以部分弥补。

---

## 二、单独组件层面的变化创新点

### 创新点 19：QK Norm 的实现差异

**Boogu-Image 的做法**：

使用 diffusers 标准的 `qk_norm="rms_norm"`，在 Attention 模块初始化时通过参数设置。

**代码依据**（`transformer_boogu.py` 第 223 行）：
```python
self.attn = Attention(
    qk_norm="rms_norm",
    eps=1e-5,
    ...)
```

**FLUX2 的做法**：

使用自定义的 `QKNorm` 类，内部包含两个带可学习 `scale` 参数的 RMSNorm。

**代码依据**（`model.py` 第 734-755 行）：
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim):
        self.scale = nn.Parameter(torch.ones(dim))  # 可学习 scale
    def forward(self, x):
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

class QKNorm(torch.nn.Module):
    def __init__(self, dim):
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)
```

**优点分析**：

- 实现差异较小。两者都对 Q 和 K 进行了 RMSNorm 归一化以稳定注意力计算。FLUX2 的 RMSNorm 带可学习 scale，Boogu-Image 使用 diffusers 的标准实现。影响不大。

---

### 创新点 20：权重初始化策略

**Boogu-Image 的做法**：

显式地对所有线性层使用 **Xavier Uniform 初始化**，对调制层（`LuminaRMSNormZero` 的 `linear`）使用 **零初始化**。

**代码依据**（`transformer_boogu.py` 第 254-267 行，989-1006 行）：
```python
nn.init.xavier_uniform_(self.attn.to_q.weight)
nn.init.xavier_uniform_(self.attn.to_k.weight)
nn.init.xavier_uniform_(self.attn.to_v.weight)
nn.init.xavier_uniform_(self.attn.to_out[0].weight)
nn.init.xavier_uniform_(self.feed_forward.linear_1.weight)
...
# 调制层零初始化
nn.init.zeros_(self.norm1.linear.weight)
nn.init.zeros_(self.norm1.linear.bias)
```

Image Index Embedding 使用正态初始化：
```python
nn.init.normal_(self.image_index_embedding, std=0.02)
```

TimestepEmbedding 也使用正态初始化：
```python
nn.init.normal_(self.linear_1.weight, std=0.02)
nn.init.zeros_(self.linear_1.bias)
```

**FLUX2 的做法**：

FLUX2 **没有显式的权重初始化代码**，依赖 PyTorch 默认初始化（`nn.Linear` 默认使用 Kaiming Uniform）。

**优点分析**：

- **训练稳定性优化**：Xavier Uniform + 调制层零初始化是经过验证的稳定初始化策略。调制参数初始化为零意味着训练开始时残差连接中的额外贡献为零（因为 `tanh(0)=0`），模型初始行为接近恒等映射，训练更稳定。这是训练稳定性维度的优化。

---

### 创新点 21：变长序列注意力（Flash Attention Varlen）支持

**Boogu-Image 的做法**：

原生支持 **变长序列的 Flash Attention**（`flash_attn_varlen_func`），通过 `unpad_input`/`pad_input` 函数处理变长序列，避免 padding 带来的无效计算。同时兼容 PyTorch 标准 `scaled_dot_product_attention` 的 fallback 实现。

**代码依据**（`attention_processor.py` 第 456-467 行）：
```python
attn_output_unpad = flash_attn_varlen_func(
    query_states, key_states, value_states,
    cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k,
    dropout_p=0.0, causal=is_causal, softmax_scale=softmax_scale,
)
```

**FLUX2 的做法**：

FLUX2 使用 PyTorch 内置的 `F.scaled_dot_product_attention`，不使用 varlen 优化。

**代码依据**（`model.py` 第 786-811 行）：
```python
attn_txt_img = F.scaled_dot_product_attention(q_txt_img, k_all, v_all, is_causal=False)
attn_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=False)
```

**优点分析**：

- **推理效率优化**：Flash Attention Varlen 支持可以在 batch 内不同样本长度不同时避免无效的 padding 计算，显著提高 GPU 利用率。这在处理不同尺寸的图像或不同长度的文本时尤为重要。这是效率维度的优化。

---

### 创新点 22：比例注意力（Proportional Attention）支持

**Boogu-Image 的做法**：

在注意力计算中支持可选的 **比例注意力缩放**（Proportional Attention），当提供 `base_sequence_length` 时，attention scale 会根据实际序列长度动态调整。

**代码依据**（`attention_processor.py` 第 1069-1074 行）：
```python
if base_sequence_length is not None:
    softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
else:
    softmax_scale = attn.scale
```

**FLUX2 的做法**：

FLUX2 **不支持** 比例注意力，使用固定的标准 scale（`head_dim ** -0.5`）。

**优点分析**：

- **生成效果优化（可变分辨率场景）**：比例注意力可以在处理不同分辨率图像时自适应调整注意力的聚焦程度，有助于模型在训练时未见过的分辨率上保持稳定的生成质量。这对于支持任意分辨率生成的场景有帮助。

---

## 三、创新点总结表

| # | 创新点 | Boogu-Image 做法 | FLUX2 做法 | 优化维度 |
|---|--------|-----------------|-----------|---------|
| 1 | 三种 Refiner 子网络 | ✅ Context/Noise/RefImage Refiner，各 2 层 | ❌ 无 | 生成效果 |
| 2 | 双流块中额外的 Image Self-Attention | ✅ Joint Attn + Image Self-Attn | ❌ 仅 Joint Attn | 生成效果 |
| 3 | 双流块独立 QKV + 独立输出投影 | ✅ img_to_q/k/v + instruct_to_q/k/v + img_out + instruct_out | ❌ 共享 qkv + 共享 proj | 生成效果 |
| 4 | 双流块 3 组调制（含 Self-Attn 调制） | ✅ img_norm1/2/3 + instruct_norm1/2 | ❌ 2 组（mod1/mod2） | 生成效果 |
| 5 | 每层独立调制 vs 全局共享调制 | ✅ 每层独立 LuminaRMSNormZero | ❌ 全局共享 3 个 Modulation | 生成效果 |
| 6 | tanh Gate 机制 | ✅ gate.tanh() * output | ❌ gate * output | 训练稳定性 |
| 7 | GQA（Grouped Query Attention） | ✅ Q=24头, KV=8头 | ❌ MHA (Q=K=V=48头) | 计算效率 |
| 8 | RMSNorm（主体归一化） | ✅ RMSNorm (Triton加速) | ❌ LayerNorm (无参数) | 计算效率 |
| 9 | SwiGLU 三线性层 FFN | ✅ linear_1 + linear_3 → SwiGLU → linear_2 | ❌ 两线性层 + chunk SiLU Gating | 表达能力(可能) |
| 10 | 注意力/FFN 输出 Post-Norm | ✅ RMSNorm after attn/FFN output | ❌ 无 | 训练稳定性 |
| 11 | 3 轴复数形式 RoPE | ✅ (40,40,40), θ=10000, 复数形式 | ❌ (32,32,32,32), θ=2000, 旋转矩阵 | 设计差异 |
| 12 | 独立参考图 Patch Embedder + Index Embedding | ✅ 独立 embedder + 5 种 index embedding | ❌ 共享 img_in + 时间坐标区分 | 生成效果 |
| 13 | 参考图原生全局注意力 | ✅ 参考图完全参与注意力交互 | ❌ 因果注意力 + KV Cache | 生成效果 vs 推理效率权衡 |
| 14 | 输出层只用 scale 调制 | ✅ LuminaLayerNormContinuous (scale only) | ❌ LastLayer (shift + scale) | 设计差异 |
| 15 | 时间步嵌入瓶颈设计 | ✅ temb 维度限制为 min(hidden,1024) | ❌ temb 维度为完整 hidden_size | 参数效率 |
| 16 | 无 Guidance Embedding | ✅ 推理管线层面实现 Double Guidance | ❌ 模型内置 guidance_in | 灵活性 vs 推理效率 |
| 17 | 更轻量的模型配置 | ✅ hidden=2304, 26层, in_ch=16 | ❌ hidden=6144, 56层, in_ch=128 | 推理效率 |
| 18 | Xavier + 零初始化调制 | ✅ 显式初始化策略 | ❌ PyTorch 默认初始化 | 训练稳定性 |
| 19 | Flash Attention Varlen | ✅ flash_attn_varlen_func | ❌ F.scaled_dot_product_attention | 计算效率 |
| 20 | 比例注意力支持 | ✅ base_sequence_length 动态 scale | ❌ 固定 scale | 可变分辨率生成效果 |

---

> **注意**：以上分析完全基于代码实现的网络结构层面，不涉及训练数据、训练策略、超参数调优等方面。实际生成效果受多方面因素影响。部分标注为"设计差异"的创新点表示两种方案各有优劣，不能简单断定哪种更好。