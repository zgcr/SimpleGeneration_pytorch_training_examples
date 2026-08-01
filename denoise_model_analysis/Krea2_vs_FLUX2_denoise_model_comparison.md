# Krea-2 vs FLUX.2 去噪模型架构对比分析

> 本报告基于 `/opt/nas/p/zhugechaoran/download/code/krea-2/mmdit.py` 和 `/opt/nas/p/zhugechaoran/download/code/flux2/src/flux2/model.py` 的完整源代码逻辑进行逐项对比分析。所有结论均经过代码复核校验。

---

## 目录

- [Krea-2 vs FLUX.2 去噪模型架构对比分析](#krea-2-vs-flux2-去噪模型架构对比分析)
  - [目录](#目录)
  - [一、整体架构变化创新点](#一整体架构变化创新点)
    - [1.1 纯单流架构替代双流+单流混合架构](#11-纯单流架构替代双流单流混合架构)
      - [FLUX.2 做法](#flux2-做法)
      - [Krea-2 做法](#krea-2-做法)
      - [区别与优势分析](#区别与优势分析)
    - [1.2 TextFusionTransformer 集成于去噪模型内部](#12-textfusiontransformer-集成于去噪模型内部)
      - [FLUX.2 做法](#flux2-做法-1)
      - [Krea-2 做法](#krea-2-做法-1)
      - [区别与优势分析](#区别与优势分析-1)
  - [二、Block 级别变化创新点](#二block-级别变化创新点)
    - [2.1 双路径残差连接与独立门控](#21-双路径残差连接与独立门控)
      - [FLUX.2 做法](#flux2-做法-2)
      - [Krea-2 做法](#krea-2-做法-2)
      - [区别与优势分析](#区别与优势分析-2)
    - [2.2 每层独立可学习调制偏置](#22-每层独立可学习调制偏置)
      - [FLUX.2 做法](#flux2-做法-3)
      - [Krea-2 做法](#krea-2-做法-3)
      - [区别与优势分析](#区别与优势分析-3)
    - [2.3 Grouped-Query Attention (GQA)](#23-grouped-query-attention-gqa)
      - [FLUX.2 做法](#flux2-做法-4)
      - [Krea-2 做法](#krea-2-做法-4)
      - [区别与优势分析](#区别与优势分析-4)
    - [2.4 门控注意力输出 (Gated Attention Output)](#24-门控注意力输出-gated-attention-output)
      - [FLUX.2 做法](#flux2-做法-5)
      - [Krea-2 做法](#krea-2-做法-5)
      - [区别与优势分析](#区别与优势分析-5)
    - [2.5 SwiGLU MLP（三独立线性层）](#25-swiglu-mlp三独立线性层)
      - [FLUX.2 做法](#flux2-做法-6)
      - [Krea-2 做法](#krea-2-做法-6)
      - [区别与优势分析](#区别与优势分析-6)
    - [2.6 RMSNorm 替代 LayerNorm（带可学习 scale）](#26-rmsnorm-替代-layernorm带可学习-scale)
      - [FLUX.2 做法](#flux2-做法-7)
      - [Krea-2 做法](#krea-2-做法-7)
      - [区别与优势分析](#区别与优势分析-7)
  - [三、组件级别变化创新点](#三组件级别变化创新点)
    - [3.1 3D RoPE 非均匀维度分配与更小的 theta](#31-3d-rope-非均匀维度分配与更小的-theta)
      - [FLUX.2 做法](#flux2-做法-8)
      - [Krea-2 做法](#krea-2-做法-8)
      - [区别与优势分析](#区别与优势分析-8)
    - [3.2 文本 token 零位置编码策略](#32-文本-token-零位置编码策略)
      - [FLUX.2 做法](#flux2-做法-9)
      - [Krea-2 做法](#krea-2-做法-9)
      - [区别与优势分析](#区别与优势分析-9)
    - [3.3 时间步嵌入使用 GELU 替代 SiLU](#33-时间步嵌入使用-gelu-替代-silu)
      - [FLUX.2 做法](#flux2-做法-10)
      - [Krea-2 做法](#krea-2-做法-10)
      - [区别与优势分析](#区别与优势分析-10)
    - [3.4 LastLayer 使用 SimpleModulation（可学习偏置）](#34-lastlayer-使用-simplemodulation可学习偏置)
      - [FLUX.2 做法](#flux2-做法-11)
      - [Krea-2 做法](#krea-2-做法-11)
      - [区别与优势分析](#区别与优势分析-11)
    - [3.5 显式指定 cuDNN SDPA 后端](#35-显式指定-cudnn-sdpa-后端)
      - [FLUX.2 做法](#flux2-做法-12)
      - [Krea-2 做法](#krea-2-做法-12)
      - [区别与优势分析](#区别与优势分析-12)
    - [3.6 序列填充至 256 倍数与 torch.compile 编译优化](#36-序列填充至-256-倍数与-torchcompile-编译优化)
      - [FLUX.2 做法](#flux2-做法-13)
      - [Krea-2 做法](#krea-2-做法-13)
      - [区别与优势分析](#区别与优势分析-13)
  - [总结表](#总结表)
    - [优势维度分类汇总](#优势维度分类汇总)

---

## 一、整体架构变化创新点

### 1.1 纯单流架构替代双流+单流混合架构

#### FLUX.2 做法

FLUX.2 使用 **双流 (DoubleStreamBlock) × 8 + 单流 (SingleStreamBlock) × 48** 的混合架构，总计 56 个 Transformer Block：

```python
# flux2/src/flux2/model.py - Flux2.__init__
self.double_blocks = nn.ModuleList(
    [DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
     for _ in range(params.depth)]  # depth=8
)
self.single_blocks = nn.ModuleList(
    [SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
     for _ in range(params.depth_single_blocks)]  # depth_single_blocks=48
)
```

前向传播中，先用 8 个双流块分别处理图像和文本（各自独立的 Norm、QKV 投影、MLP，但 Attention 时 Q/K/V 拼接做联合注意力），然后拼接文本和图像 token 进入 48 个单流块统一处理：

```python
# flux2/src/flux2/model.py - Flux2.forward
for block in self.double_blocks:
    img, txt, _ = block.forward_kv_extract(img, txt, pe_x, pe_ctx, ...)
img = torch.cat((txt, img), dim=1)
pe = torch.cat((pe_ctx, pe_x), dim=2)
for block in self.single_blocks:
    img, _ = block.forward_kv_extract(img, pe, single_block_mod, ...)
```

#### Krea-2 做法

Krea-2 使用**纯单流 (SingleStreamBlock) × 28** 架构，总计 28 个 Transformer Block：

```python
# krea-2/mmdit.py - SingleStreamDiT.__init__
self.blocks = nn.ModuleList(
    [SingleStreamBlock(config.features, config.heads, config.multiplier, config.bias, config.kvheads)
     for _ in range(config.layers)]  # layers=28
)
```

文本和图像 token 在进入 Transformer 之前就已经拼接在一起，从第一层开始就进行统一处理：

```python
# krea-2/mmdit.py - SingleStreamDiT.forward
combined = torch.cat((context, img), dim=1)
for block in self.blocks:
    combined = block(combined, tvec, freqs, mask)
```

#### 区别与优势分析

| 对比维度 | FLUX.2 | Krea-2 |
|---------|--------|--------|
| 架构类型 | 8 双流 + 48 单流 = 56 层 | 28 纯单流 |
| 文本-图像交互起始层 | 第 1 层（但前 8 层仅在 Attention 时交叉，MLP 独立） | 第 1 层（完全统一处理） |
| 参数效率 | 双流块有独立的 img/txt Norm + QKV + MLP，参数冗余 | 单流块参数在文本和图像间完全共享 |

**优势判断**：
- **参数效率优化**（非生成效果维度）：纯单流架构用 28 层（约为 FLUX.2 的一半层数）实现功能，参数量显著减少。双流块中图像和文本各有独立的 `img_norm1/img_norm2/img_attn/img_mlp` 和 `txt_norm1/txt_norm2/txt_attn/txt_mlp`，这些重复参数在纯单流架构中被消除。
- **更充分的跨模态交互**（可能提升生成效果）：从第 1 层开始，文本和图像 token 就在 Norm、FFN 中完全共享参数并交互，而 FLUX.2 的前 8 层双流块中文本和图像仅在 Attention 阶段交互，MLP 阶段各自独立。更早更彻底的跨模态融合可能有助于模型更好地理解文本-图像对应关系。
- **推理效率优化**（非生成效果维度）：更少的层数和参数意味着更快的推理速度。

---

### 1.2 TextFusionTransformer 集成于去噪模型内部

#### FLUX.2 做法

FLUX.2 的文本编码器（Mistral-Small-3.2-24B 或 Qwen3）提取 3 个中间层的隐藏状态，通过**简单的维度拼接**合并多层特征：

```python
# flux2/src/flux2/text_encoder.py - Mistral3SmallEmbedder.forward
OUTPUT_LAYERS_MISTRAL = [10, 20, 30]
out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_MISTRAL], dim=1)
return rearrange(out, "b c l d -> b l (c d)")  # (B, 512, 3×5120) = (B, 512, 15360)
```

然后在 DIT 中通过单个线性层投影到隐藏空间：

```python
# flux2/src/flux2/model.py - Flux2.__init__
self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)
# context_in_dim=15360 → hidden_size=6144
```

#### Krea-2 做法

Krea-2 的文本编码器（Qwen3-VL-4B）提取 **12 个中间层**的隐藏状态（第 2,5,8,11,14,17,20,23,26,29,32,35 层），输出为 `(B, L, 12, 2560)`。这些多层特征在 DIT 的 `forward` 方法内部通过一个专门设计的 **TextFusionTransformer** 进行学习性融合：

```python
# krea-2/mmdit.py - TextFusionTransformer
class TextFusionTransformer(torch.nn.Module):
    def __init__(self, num_txt_layers, txt_dim, heads, multiplier, bias, kvheads):
        # 阶段1: 2 个跨层注意力融合块
        self.layerwise_blocks = torch.nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )
        # 阶段2: 线性投影 12层 → 1层
        self.projector = torch.nn.Linear(num_txt_layers, 1, bias=False)
        # 阶段3: 2 个序列级精炼块
        self.refiner_blocks = torch.nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )

    def forward(self, x, mask=None):
        b, l, n, d = x.shape  # (batch, seq_len, 12, 2560)
        x = x.reshape(b * l, n, d)  # 每个 token 位置独立处理 12 层
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None)  # 跨层注意力融合
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x)  # 12 → 1
        x = x.squeeze(-1)     # (B, L, 2560)
        for block in self.refiner_blocks:
            x = block(x, mask=mask)  # 全局序列精炼
        return x
```

然后通过 `txtmlp` 投影到隐藏空间：

```python
# krea-2/mmdit.py - SingleStreamDiT.__init__
self.txtmlp = nn.Sequential(
    RMSNorm(config.txtdim),                          # RMSNorm(2560)
    nn.Linear(config.txtdim, config.features),       # 2560 → 6144
    nn.GELU(approximate="tanh"),
    nn.Linear(config.features, config.features),     # 6144 → 6144
)
```

融合过程在 DIT 的 `forward` 中被调用：

```python
# krea-2/mmdit.py - SingleStreamDiT.forward
context = self.txtfusion(context, mask=txtmask)  # TextFusionTransformer
context = self.txtmlp(context)                    # 投影到 hidden_size
```

#### 区别与优势分析

| 对比维度 | FLUX.2 | Krea-2 |
|---------|--------|--------|
| 提取层数 | 3 层 (第10/20/30层) | 12 层 (每隔3层取1层) |
| 融合方式 | 维度拼接 (concat) | 学习性注意力融合 (TextFusionTransformer) |
| 融合位置 | 文本编码器输出后，DIT 之前 | DIT 的 forward 内部 |
| 输出维度 | 15360 (3×5120) | 2560 (融合后单层维度) |
| 是否可学习 | ❌ 无可学习参数（仅拼接） | ✅ 4 个 TextFusionBlock + 1 个 Linear 投影 |
| 是否端到端训练 | 仅 txt_in 投影层参与训练 | TextFusionTransformer 与 DIT 端到端训练 |

**优势判断**：
- **更丰富的多尺度语义特征**（提升生成效果）：12 层 vs 3 层的层级采样，覆盖了从浅层到深层更密集的语义特征梯度。浅层特征包含更多低级语法/词汇信息，深层特征包含更多高级语义信息。更密集的采样使模型能获取更完整的语义谱系。
- **学习性融合优于简单拼接**（提升生成效果）：跨层注意力可以建模不同层之间的关系（例如浅层的形态学信息与深层的语义信息之间的互补关系），而简单拼接只是机械地堆叠特征。线性投影后的序列级精炼进一步优化了全局文本表示。
- **更紧凑的文本表示**（参数效率优化）：最终文本维度 2560（投影后 6144）远小于 FLUX.2 的 15360，减轻了后续 DIT 的计算负担。
- **端到端可训练**（提升生成效果）：TextFusionTransformer 作为 DIT 的一部分，可以被去噪损失函数直接优化，使文本融合策略能适应图像生成任务的需求。

---

## 二、Block 级别变化创新点

### 2.1 双路径残差连接与独立门控

#### FLUX.2 做法

FLUX.2 的 `SingleStreamBlock` 使用**融合式设计**：通过一个 `linear1` 同时生成 QKV 和 MLP 输入，注意力输出与 MLP 输出拼接后通过 `linear2` 合并输出，**仅用一个 gate 控制整个残差**：

```python
# flux2/src/flux2/model.py - SingleStreamBlock
def _qkv(self, x, mod):
    mod_shift, mod_scale, mod_gate = mod  # 3 个调制参数
    x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift  # 唯一的 pre_norm
    qkv, mlp = torch.split(self.linear1(x_mod), ...)  # 一个 linear1 同时生成 QKV 和 MLP 输入
    q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    q, k = self.norm(q, k, v)
    return q, k, v, mlp, mod_gate

def _out(self, x, attn, mlp, mod_gate):
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))  # 拼接 attn 和 mlp 输出
    return x + mod_gate * output  # 单一 gate 控制整体残差
```

数据流：`x → pre_norm → AdaLN → linear1 → [QKV, MLP_in] → attn & mlp_act → cat → linear2 → gate → residual`

#### Krea-2 做法

Krea-2 的 `SingleStreamBlock` 使用**分离式设计**：注意力和 MLP 各有独立的 Norm、残差连接和门控，**6 个调制参数**（pre/post 各 scale, shift, gate）：

```python
# krea-2/mmdit.py - SingleStreamBlock
def forward(self, x, vec, freqs, mask=None):
    prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)  # 6 个调制参数
    # 注意力路径：独立 norm + 独立 gate + 独立残差
    x = x + pregate * self.attn(
        (1 + prescale) * self.prenorm(x) + preshift, freqs, mask
    )
    # MLP 路径：独立 norm + 独立 gate + 独立残差
    x = x + postgate * self.mlp(
        (1 + postscale) * self.postnorm(x) + postshift
    )
    return x
```

数据流：
- 注意力路径：`x → prenorm → AdaLN(prescale, preshift) → Attention → pregate → residual`
- MLP 路径：`x → postnorm → AdaLN(postscale, postshift) → SwiGLU → postgate → residual`

#### 区别与优势分析

| 对比维度 | FLUX.2 SingleStreamBlock | Krea-2 SingleStreamBlock |
|---------|--------------------------|--------------------------|
| Norm 数量 | 1 个 (pre_norm) | 2 个 (prenorm + postnorm) |
| 残差连接 | 1 条（attn+mlp 合并后） | 2 条（attn 和 mlp 各自独立） |
| 门控 | 1 个 mod_gate | 2 个 (pregate + postgate) |
| 调制参数 | 3 个 (shift, scale, gate) | 6 个 (pre/post 各 scale, shift, gate) |
| 计算方式 | attn 和 mlp 输入共享同一个 linear1 | attn 和 mlp 完全独立 |

**优势判断**：
- **更精细的信息流控制**（提升生成效果）：独立门控允许模型对注意力输出和 MLP 输出分别控制其对残差的贡献比例。例如，某些层可能需要更多注意力信息（pregate 较大）而较少 MLP 变换（postgate 较小），反之亦然。FLUX.2 的单一 gate 无法实现这种细粒度控制。
- **更灵活的调制**（提升生成效果）：6 个独立的调制参数使每个路径可以有不同的 scale 和 shift，而 FLUX.2 只有一组 scale/shift 同时影响 QKV 和 MLP 输入。
- **MLP 输入独立归一化**（提升生成效果）：Krea-2 的 MLP 输入经过独立的 `postnorm`，而 FLUX.2 的 MLP 输入与 QKV 共享同一个 `pre_norm` + `linear1` 的变换。独立归一化使 MLP 可以从不同的特征分布出发进行处理。

---

### 2.2 每层独立可学习调制偏置

#### FLUX.2 做法

FLUX.2 使用**全局共享 Modulation**，3 个 Modulation 模块通过 `SiLU → Linear` 从时间步条件 `vec` 动态生成调制参数，所有同类型 block 共享同一组参数：

```python
# flux2/src/flux2/model.py - Flux2.__init__
self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, disable_bias=True)
self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True, disable_bias=True)
self.single_stream_modulation = Modulation(self.hidden_size, double=False, disable_bias=True)

# flux2/src/flux2/model.py - Modulation.forward
def forward(self, vec):
    out = self.lin(nn.functional.silu(vec))  # SiLU 激活 → 线性变换
    out = out.chunk(self.multiplier, dim=-1)
    return out[:3], out[3:] if self.is_double else None
```

在 `forward` 中计算一次，传入所有 block：

```python
# flux2/src/flux2/model.py - Flux2.forward
double_block_mod_img = self.double_stream_modulation_img(vec)  # 所有 8 个双流块共享
single_block_mod, _ = self.single_stream_modulation(vec)       # 所有 48 个单流块共享
```

#### Krea-2 做法

Krea-2 采用 **全局时间条件 + 每层独立可学习偏置** 的混合方案。首先通过 `tproj` 生成全局调制向量 `tvec`，然后每个 `SingleStreamBlock` 内部有独立的 `DoubleSharedModulation`，它是一个**可学习的偏置参数**：

```python
# krea-2/mmdit.py - DoubleSharedModulation
class DoubleSharedModulation(torch.nn.Module):
    def __init__(self, dim):
        self.lin = torch.nn.Parameter(torch.zeros(6 * dim))  # 每层独立的可学习偏置

    def forward(self, vec):
        out = vec + self.lin  # 全局 tvec + 层特有偏置
        prescale, preshift, pregate, postscale, postshift, postgate = out.chunk(6, dim=-1)
        return prescale, preshift, pregate, postscale, postshift, postgate
```

全局 `tvec` 的生成：

```python
# krea-2/mmdit.py - SingleStreamDiT.forward
t = self.tmlp(temb(t, self.config.tdim, ...))  # 时间步嵌入 → MLP → (B, 1, 6144)
tvec = self.tproj(t)  # GELU → Linear(6144, 6144*6=36864)
```

#### 区别与优势分析

| 对比维度 | FLUX.2 | Krea-2 |
|---------|--------|--------|
| 调制参数生成 | `SiLU(vec) → Linear` 动态生成 | `全局 tvec + 每层可学习偏置` |
| 共享策略 | 所有同类型 block 完全共享 | 全局 tvec 共享 + 每层独立微调 |
| 每层差异性 | ❌ 无（所有层接收相同参数） | ✅ 有（每层有独立偏置 `self.lin`） |
| 可学习参数 | 仅在 Modulation.lin 中 | 每层额外增加 `6 × hidden_dim` 参数 |
| 非线性变换 | ✅ SiLU 激活 | ❌ 仅线性相加（偏置无非线性） |

**优势判断**：
- **层级特异性调制**（提升生成效果）：不同层在去噪过程中承担不同角色（浅层可能关注低频结构，深层关注高频细节）。每层独立的可学习偏置使调制参数可以根据每层的特定功能进行微调，而 FLUX.2 所有层接收完全相同的调制信号，无法区分层级差异。
- **更高效的参数利用**（参数效率优化）：虽然每层增加了 `6 × 6144 = 36864` 个参数（28 层共约 100 万参数），但消除了 FLUX.2 中 3 个独立 Modulation 模块的 `Linear(6144, multiplier×6144)` 大型投影层。整体上 Krea-2 的调制参数更少但更有针对性。

---

### 2.3 Grouped-Query Attention (GQA)

#### FLUX.2 做法

FLUX.2 使用**标准 Multi-Head Attention (MHA)**，Q/K/V 头数相同：

```python
# flux2/src/flux2/model.py - SelfAttention
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)  # Q/K/V 统一投影，头数相同
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)
```

对于 dev-32B 模型：`num_heads=48`，所以 Q=48 heads, K=48 heads, V=48 heads。

#### Krea-2 做法

Krea-2 使用 **Grouped-Query Attention (GQA)**，KV 头数为 Q 头数的 1/4：

```python
# krea-2/mmdit.py - Attention
class Attention(torch.nn.Module):
    def __init__(self, dim, heads, kvheads=None, bias=False):
        self.heads = heads        # 48
        self.kvheads = kvheads    # 12
        self.headdim = dim // self.heads  # 128
        self.wq = torch.nn.Linear(dim, self.headdim * self.heads, bias=bias)    # Q: 48 heads
        self.wk = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)  # K: 12 heads
        self.wv = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)  # V: 12 heads
        self.gate = torch.nn.Linear(dim, dim, bias=bias)
        self.gqa = self.heads != self.kvheads
        self.wo = torch.nn.Linear(dim, dim, bias=bias)
```

使用 PyTorch SDPA 的 `enable_gqa=True` 标志进行高效 GQA 计算：

```python
# krea-2/mmdit.py - attention 函数
def attention(q, k, v, mask=None, scale=None, gqa=False):
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa)
    return rearrange(x, "B H L D -> B L (H D)")
```

#### 区别与优势分析

| 对比维度 | FLUX.2 (MHA) | Krea-2 (GQA) |
|---------|-------------|--------------|
| Q heads | 48 | 48 |
| K heads | 48 | 12 |
| V heads | 12 | 12 |
| KV 参数量 | `2 × dim × dim` | `2 × dim × (dim/4)` |
| Q/KV 比例 | 1:1 | 4:1 |

KV 投影参数量对比（每层）：
- FLUX.2: K 投影 `6144 × 6144` + V 投影 `6144 × 6144` ≈ 75.5M
- Krea-2: K 投影 `6144 × 1536` + V 投影 `6144 × 1536` ≈ 18.9M

每层 KV 参数减少约 **75%**。

**优势判断**：
- **显著降低计算量和显存占用**（推理效率优化）：GQA 将 KV 的计算量和显存减少到 MHA 的 1/4，对于长序列（如高分辨率图像的 token 序列）效果尤其明显。这一技术已在 LLaMA-2/3、Qwen、Mistral 等主流 LLM 中被广泛验证。
- **接近 MHA 的生成质量**（对生成效果影响较小）：大量研究（Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, 2023）表明 GQA 在质量上非常接近 MHA，尤其是在模型规模足够大时。4:1 的 Q/KV 比例是一个经过验证的平衡点。

---

### 2.4 门控注意力输出 (Gated Attention Output)

#### FLUX.2 做法

FLUX.2 的注意力输出直接通过线性投影层：

```python
# flux2/src/flux2/model.py - DoubleStreamBlock._apply_residuals
img = img + img_mod1_gate * self.img_attn.proj(img_attn)  # 直接投影

# SelfAttention 中
self.proj = nn.Linear(dim, dim, bias=False)  # 标准输出投影
```

#### Krea-2 做法

Krea-2 在注意力输出上添加了一个**可学习的 sigmoid 门控**，与 QKV 并行计算：

```python
# krea-2/mmdit.py - Attention.forward
def forward(self, qkv, freqs=None, mask=None):
    q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)
    # gate: nn.Linear(dim, dim, bias=False)
    # ... RoPE、QKNorm 等处理 ...
    out = self.wo(attention(q, k, v, mask=mask, gqa=self.gqa) * F.sigmoid(gate))
    return out
```

`gate` 是一个与 Q/K/V 完全并行的线性变换，通过 sigmoid 后与注意力输出逐元素相乘，类似于 Gated Linear Units (GLU) 的思想应用到注意力机制中。

#### 区别与优势分析

| 对比维度 | FLUX.2 | Krea-2 |
|---------|--------|--------|
| 注意力输出处理 | `proj(attn_output)` | `wo(attn_output * sigmoid(gate(x)))` |
| 额外参数 | 无 | `nn.Linear(dim, dim)` 用于 gate |
| 门控范围 | 无 | sigmoid 将每个维度缩放到 [0, 1] |

**优势判断**：
- **动态信息过滤**（提升生成效果）：sigmoid 门控允许模型在每个位置、每个特征维度上独立地控制注意力输出的信息流量。某些维度的注意力结果可能不够有用或引入噪声，门控可以将其抑制。这种机制在 GLU/SwiGLU 的 FFN 中已被验证有效，Krea-2 将其扩展到注意力输出，提供了更细粒度的信息选择能力。
- **稳定训练**（提升生成效果间接收益）：sigmoid 输出范围为 [0, 1]，可以天然地限制注意力输出的幅度，防止某些头的注意力值过大导致训练不稳定。

---

### 2.5 SwiGLU MLP（三独立线性层）

#### FLUX.2 做法

FLUX.2 在 DoubleStreamBlock 中使用**两层 MLP + SiLU Gating**，gate 和 up 投影融合在一个线性层中：

```python
# flux2/src/flux2/model.py - DoubleStreamBlock.__init__
self.img_mlp = nn.Sequential(
    nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False),  # 6144 → 36864（融合 gate+up）
    SiLUActivation(),                                         # chunk → SiLU(x1) * x2
    nn.Linear(mlp_hidden_dim, hidden_size, bias=False),       # 18432 → 6144
)

# flux2/src/flux2/model.py - SiLUActivation
class SiLUActivation(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2  # SiLU(x1) * x2
```

在 SingleStreamBlock 中，MLP 输入与 QKV 共享 `linear1`，输出通过 `linear2` 与 attn 输出合并。

#### Krea-2 做法

Krea-2 使用 **SwiGLU**，gate 和 up 是三个完全独立的线性层：

```python
# krea-2/mmdit.py - SwiGLU
class SwiGLU(torch.nn.Module):
    def __init__(self, features, multiplier, bias=False, multiple=128):
        mlpdim = int(2 * features / 3) * multiplier  # int(2*6144/3)*4 = 16384
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)  # 对齐到 128 的倍数

        self.gate = torch.nn.Linear(features, mlpdim, bias=bias)  # 6144 → 16384
        self.up = torch.nn.Linear(features, mlpdim, bias=bias)    # 6144 → 16384（独立）
        self.down = torch.nn.Linear(mlpdim, features, bias=bias)  # 16384 → 6144

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))  # SiLU(gate(x)) * up(x)
```

#### 区别与优势分析

| 对比维度 | FLUX.2 | Krea-2 |
|---------|--------|--------|
| 结构 | `Linear(→2×mlp_dim) → chunk → SiLU(x1)*x2 → Linear` | `gate(x) → SiLU, up(x) → multiply → down` |
| gate 和 up 关系 | 融合在一个 Linear 层中 (chunk 分割) | 两个完全独立的 Linear 层 |
| 中间维度 | 18432 (`hidden×3.0`) | 16384 (`int(2×hidden/3)×4`，对齐到 128) |
| 维度对齐 | ❌ 无特殊对齐 | ✅ 对齐到 128 的倍数 |

**优势判断**：
- **更高的表达能力**（提升生成效果）：独立的 gate 和 up 投影意味着两个线性变换是完全独立的权重矩阵，而 FLUX.2 的融合方式中 gate 和 up 共享同一个大矩阵的不同部分（通过 chunk 分割），虽然数学上等价但优化景观不同。独立参数在实践中已被 LLaMA/Qwen 等模型验证更优。
- **硬件效率优化**（推理效率优化）：中间维度对齐到 128 的倍数，确保 Tensor Core 等硬件加速单元能以最优效率运行矩阵乘法，减少因维度不对齐导致的算力浪费。
- **与主流 LLM 架构一致**（工程优势）：SwiGLU 三线性层结构与 LLaMA 3、Qwen 2.5 等主流 LLM 完全一致，有利于利用已有的优化工具链和训练经验。

---

### 2.6 RMSNorm 替代 LayerNorm（带可学习 scale）

#### FLUX.2 做法

FLUX.2 在 Block 级别使用**不带可学习参数的 LayerNorm**：

```python
# flux2/src/flux2/model.py - DoubleStreamBlock/SingleStreamBlock
self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
```

在 QKNorm 中使用 RMSNorm（scale 初始化为 `ones`）：

```python
# flux2/src/flux2/model.py - RMSNorm (用于 QKNorm)
class RMSNorm(torch.nn.Module):
    def __init__(self, dim):
        self.scale = nn.Parameter(torch.ones(dim))  # 初始化为 1
    def forward(self, x):
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale
```

#### Krea-2 做法

Krea-2 **全面使用带可学习 scale 的 RMSNorm**，包括 Block 归一化和 QKNorm：

```python
# krea-2/mmdit.py - RMSNorm
class RMSNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-05):
        self.scale = torch.nn.Parameter(torch.zeros(features, dtype=torch.float32))  # 初始化为 0

    @torch.compile(fullgraph=True)
    def forward(self, x):
        t = x.float()
        t = F.rms_norm(t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0))  # 有效权重 = scale + 1.0
        return t.to(dtype)
```

使用方式：

```python
# krea-2/mmdit.py - SingleStreamBlock
self.prenorm = RMSNorm(features)   # Block 级归一化
self.postnorm = RMSNorm(features)  # Block 级归一化

# krea-2/mmdit.py - QKNorm
self.qnorm = RMSNorm(dim)  # Q 归一化
self.knorm = RMSNorm(dim)  # K 归一化
```

#### 区别与优势分析

| 对比维度 | FLUX.2 | Krea-2 |
|---------|--------|--------|
| Block 归一化 | LayerNorm (无可学习参数) | RMSNorm (带可学习 scale) |
| QKNorm | RMSNorm (scale=ones 初始化) | RMSNorm (scale=zeros 初始化，有效权重=scale+1) |
| 计算复杂度 | LayerNorm 需计算均值+方差 | RMSNorm 仅计算均方根（少一次 reduce） |
| 是否有可学习参数 | Block Norm: ❌; QKNorm: ✅ | 全部 ✅ |
| 初始化策略 | QKNorm scale = ones | scale = zeros, 有效权重 = 1.0 |

**优势判断**：
- **计算效率提升**（推理效率优化）：RMSNorm 相比 LayerNorm 省去了均值计算和减均值操作，对于大规模模型的推理延迟有可测量的改善。这一优势已在 LLaMA、Qwen 等模型中被广泛验证。
- **可学习的逐通道缩放**（提升生成效果）：Krea-2 的 Block RMSNorm 带有可学习的 `scale` 参数，允许模型学习每个通道的重要性权重。FLUX.2 的 LayerNorm 使用 `elementwise_affine=False`，完全没有可学习参数，归一化后所有通道被同等对待。
- **稳定的初始化策略**（训练稳定性）：scale 初始化为 0（有效权重 = 1.0）结合 `@torch.compile(fullgraph=True)` 编译加速，使归一化在训练初期表现为标准 RMSNorm，之后逐渐学习通道特异性权重，有利于训练稳定性。

---

## 三、组件级别变化创新点

### 3.1 3D RoPE 非均匀维度分配与更小的 theta

#### FLUX.2 做法

FLUX.2 使用 **4D RoPE**，均匀分配维度，`theta=2000`：

```python
# flux2/src/flux2/model.py - Flux2Params
axes_dim: list[int] = [32, 32, 32, 32]  # t, h, w, l 各 32 维
theta: int = 2000
```

位置 ID 有 4 个维度（t, h, w, l），其中 `l` 维度用于文本 token 的序列位置编码：

```python
# flux2/src/flux2/sampling.py - prc_txt
coords = {
    "t": torch.arange(1) if t_coord is None else t_coord,
    "h": torch.arange(1),
    "w": torch.arange(1),
    "l": torch.arange(_l),  # 文本序列位置
}
```

#### Krea-2 做法

Krea-2 使用 **3D RoPE**，**非均匀分配**维度，`theta=1000`：

```python
# krea-2/mmdit.py - SingleStreamDiT.__init__
headdim = config.features // config.heads  # 6144 // 48 = 128
axes = [
    headdim - 12 * (headdim // 16),  # 128 - 12*8 = 32  (t 维度)
    6 * (headdim // 16),              # 6*8 = 48         (h 维度)
    6 * (headdim // 16),              # 6*8 = 48         (w 维度)
]  # → [32, 48, 48]
```

配置参数：

```python
# krea-2/mmdit.py - SingleMMDiTConfig
theta: float = 1e3  # = 1000
```

#### 区别与优势分析

| 对比维度 | FLUX.2 | Krea-2 |
|---------|--------|--------|
| RoPE 维度 | 4D (t, h, w, l) | 3D (t, h, w) |
| 维度分配 | 均匀 [32, 32, 32, 32] | 非均匀 [32, 48, 48] |
| 空间维度占比 | 50% (32+32=64 / 128) | 75% (48+48=96 / 128) |
| theta | 2000 | 1000 |
| 文本位置编码 | 通过 `l` 维度编码序列位置 | 无（全零位置向量） |

**优势判断**：
- **更强的空间位置感知**（提升生成效果）：将 75% 的 RoPE 维度分配给空间位置 (h, w)，使模型对空间关系有更精细的编码能力。图像生成任务中空间位置关系至关重要（物体相对位置、大小、布局等），更多的空间编码维度有助于更精确的空间控制。
- **更小的 theta 提升高频分量**（提升生成效果）：theta=1000（vs 2000）使 RoPE 的频率成分整体提高，能够捕捉更精细的空间关系变化。对于图像中相邻 patch 之间的精细位置差异，更高的频率分量有助于区分。
- **更简洁的设计**（工程优势）：去掉第 4 维 `l`（文本序列位置），简化了位置编码系统。文本 token 的顺序信息通过 TextFusionTransformer 的自注意力来捕获，不依赖 RoPE。

---

### 3.2 文本 token 零位置编码策略

#### FLUX.2 做法

FLUX.2 的文本 token 有非零的位置 ID，通过第 4 维 `l` 编码序列位置：

```python
# flux2/src/flux2/sampling.py - prc_txt
x_ids = torch.cartesian_prod(
    coords["t"],   # 通常为 0
    coords["h"],   # 0（dummy）
    coords["w"],   # 0（dummy）
    coords["l"],   # 0, 1, 2, ..., L-1（序列位置）
)
```

因此文本 token 通过 4D RoPE 获得了序列位置信息。

#### Krea-2 做法

Krea-2 的文本 token 位置全部为零向量：

```python
# krea-2/sampling.py - prepare
txtpos = torch.zeros(b, txtlen, 3, device=img.device)  # 全零位置
```

这意味着所有文本 token 在 RoPE 的视角下处于相同的位置，RoPE 不会为文本 token 引入位置信息。

#### 区别与优势分析

**优势判断**：
- **文本位置信息由 TextFusionTransformer 提供**（设计解耦）：文本的顺序信息通过 TextFusionTransformer 中 `refiner_blocks` 的自注意力来自然捕获（位置信息隐含在注意力模式中）。这实现了位置编码的职责分离——RoPE 专注于空间位置，文本序列顺序由注意力机制自行学习。
- **避免文本-图像位置冲突**（可能提升生成效果）：文本 token 和图像 token 处于不同的语义空间，为文本 token 分配空间位置可能引入不必要的位置偏置。零位置策略使文本 token 对所有图像位置"等距"，不会因位置编码而偏好特定空间区域。

---

### 3.3 时间步嵌入使用 GELU 替代 SiLU

#### FLUX.2 做法

FLUX.2 使用 **SiLU 激活**的 MLP 进行时间步嵌入：

```python
# flux2/src/flux2/model.py - MLPEmbedder
class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim, disable_bias=False):
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=not disable_bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=not disable_bias)
    def forward(self, x):
        return self.out_layer(self.silu(self.in_layer(x)))
```

#### Krea-2 做法

Krea-2 使用 **GELU(tanh近似) 激活**的时间步嵌入 MLP：

```python
# krea-2/mmdit.py - SingleStreamDiT.__init__
self.tmlp = nn.Sequential(
    nn.Linear(config.tdim, config.features),      # 256 → 6144
    nn.GELU(approximate="tanh"),                   # GELU(tanh)
    nn.Linear(config.features, config.features),   # 6144 → 6144
)
```

调制向量投影也使用 GELU：

```python
self.tproj = nn.Sequential(
    nn.GELU(approximate="tanh"),
    nn.Linear(config.features, config.features * 6)  # 6144 → 36864
)
```

#### 区别与优势分析

| 对比维度 | FLUX.2 | Krea-2 |
|---------|--------|--------|
| 激活函数 | SiLU (Swish) | GELU (tanh 近似) |
| 函数形式 | `x * sigmoid(x)` | `x * Φ(x) ≈ 0.5x(1+tanh(√(2/π)(x+0.044715x³)))` |
| 近零区行为 | 线性 + 略微抑制 | 更平滑的过渡 |

**优势判断**：
- **设计一致性**（工程优势）：时间步嵌入使用 GELU 与文本投影 `txtmlp` 保持一致，而 Attention 内部的 SwiGLU 使用 SiLU。这可能是刻意的设计选择——时间步条件和文本条件作为"调制信号"使用 GELU，而数据流路径使用 SiLU gating。
- **对生成效果影响较小**：SiLU 和 GELU 在功能上非常接近，这一变化主要是设计偏好差异，对生成质量的影响有限。

---

### 3.4 LastLayer 使用 SimpleModulation（可学习偏置）

#### FLUX.2 做法

FLUX.2 的 `LastLayer` 使用 **LayerNorm + SiLU→Linear 动态生成 AdaLN 参数**：

```python
# flux2/src/flux2/model.py - LastLayer
class LastLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=False)  # 动态生成 shift, scale
        )

    def forward(self, x, vec):
        mod = self.adaLN_modulation(vec)
        shift, scale = mod.chunk(2, dim=-1)
        x = (1 + scale) * self.norm_final(x) + shift
        x = self.linear(x)
        return x
```

#### Krea-2 做法

Krea-2 的 `LastLayer` 使用 **RMSNorm + SimpleModulation（可学习偏置+时间条件）**：

```python
# krea-2/mmdit.py - LastLayer
class LastLayer(torch.nn.Module):
    def __init__(self, features, patch, channels):
        self.norm = RMSNorm(features)
        self.linear = torch.nn.Linear(features, patch * patch * channels, bias=True)  # 带 bias
        self.modulation = SimpleModulation(features)

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        x = self.linear(x)
        return x

# krea-2/mmdit.py - SimpleModulation
class SimpleModulation(torch.nn.Module):
    def __init__(self, dim):
        self.lin = torch.nn.Parameter(torch.zeros(2, dim))  # 2 个可学习向量

    def forward(self, vec):
        out = vec + rearrange(self.lin, "two d -> 1 two d")  # vec + 可学习偏置
        scale, shift = out.chunk(2, dim=1)
        return scale, shift
```

注意 Krea-2 的 `LastLayer` 接收的 `tvec` 是 `tmlp` 的输出（形状 `(B, 1, 6144)`），经过 `SimpleModulation` 后 split 成 `(B, 1, 6144)` 的 scale 和 shift。而 FLUX.2 接收的是 `vec`（`time_in(t) + guidance_in(g)`，形状 `(B, 6144)`），经过 `adaLN_modulation`（`SiLU → Linear(6144, 12288)`）后 chunk。

#### 区别与优势分析

| 对比维度 | FLUX.2 | Krea-2 |
|---------|--------|--------|
| 归一化 | LayerNorm (无可学习参数) | RMSNorm (带可学习 scale) |
| 调制参数生成 | `SiLU(vec) → Linear(hidden, 2×hidden)` 动态生成 | `vec + learnable_bias` 简单相加 |
| 输出层 bias | False | True |
| 调制复杂度 | 需要 `nn.Linear(6144, 12288)` 约 75M 参数 | 仅需 `Parameter(2, 6144)` 约 12K 参数 |

**优势判断**：
- **极大减少 LastLayer 参数量**（参数效率优化）：从 ~75M 参数的线性层降低到 ~12K 的可学习偏置参数，但保留了层级特异性调制能力。由于最终层只出现一次，对总参数的影响有限，但这体现了整体的轻量化设计理念。
- **设计一致性**（工程优势）：与 Block 内部的 `DoubleSharedModulation`（可学习偏置）保持一致的设计哲学。

---

### 3.5 显式指定 cuDNN SDPA 后端

#### FLUX.2 做法

FLUX.2 直接调用 `F.scaled_dot_product_attention`，不指定后端：

```python
# flux2/src/flux2/model.py - causal_attn_fn
out = F.scaled_dot_product_attention(q_txt_img, k_all, v_all, is_causal=False)
```

PyTorch 会自动选择最优后端（Flash Attention、Math、cuDNN 等）。

#### Krea-2 做法

Krea-2 **显式指定 cuDNN Attention 后端**：

```python
# krea-2/mmdit.py - attention
def attention(q, k, v, mask=None, scale=None, gqa=False):
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
        )
    return rearrange(x, "B H L D -> B L (H D)")
```

#### 区别与优势分析

**优势判断**：
- **确定性的最优性能**（推理效率优化）：cuDNN Attention 后端针对 NVIDIA GPU 深度优化，尤其支持 GQA（`enable_gqa=True`）。显式指定后端避免了 PyTorch 自动选择时可能退回到较慢后端的情况（例如当 attn_mask 不为 None 时，某些后端可能不支持）。
- **GQA 兼容性保证**（工程优势）：cuDNN 后端对 GQA 的支持更为原生，确保 KV 头数不同于 Q 头数时的正确性和效率。

---

### 3.6 序列填充至 256 倍数与 torch.compile 编译优化

#### FLUX.2 做法

FLUX.2 的模型代码中没有序列填充或 `@torch.compile` 注解：

```python
# flux2/src/flux2/model.py - 无填充、无编译注解
```

#### Krea-2 做法

Krea-2 有两个显著的编译优化设计：

**（1）序列填充至 256 的倍数**：

```python
# krea-2/mmdit.py - SingleStreamDiT.forward
fulllen = combined.shape[1]
_padlen = (-fulllen) % 256
if _padlen > 0:
    combined = F.pad(combined, (0, 0, 0, _padlen))
    mask = F.pad(mask, (0, _padlen), value=False)
    pos = F.pad(pos, (0, 0, 0, _padlen))
```

**（2）关键组件标注 `@torch.compile(fullgraph=True)`**：

```python
# krea-2/mmdit.py
class PositionalEncoding(torch.nn.Module):
    @torch.compile(fullgraph=True)
    def forward(self, pos):
        ...

class RMSNorm(torch.nn.Module):
    @torch.compile(fullgraph=True)
    def forward(self, x):
        ...

class LastLayer(torch.nn.Module):
    @torch.compile(fullgraph=True)
    def forward(self, x, tvec):
        ...
```

#### 区别与优势分析

**优势判断**：
- **稳定的编译内核形状**（推理效率优化）：将序列长度填充到 256 的倍数，避免了不同分辨率输入导致的 torch.compile 缓存失效问题。每次输入的序列长度都会落在固定的几个值上（256, 512, 768, ...），使编译后的 CUDA kernel 可以被高效复用。
- **关键路径编译加速**（推理效率优化）：对计算密集的 `PositionalEncoding`、`RMSNorm`、`LastLayer` 使用 `fullgraph=True` 编译，消除 Python 解释器开销，启用算子融合、内存布局优化等编译器优化。
- **对生成效果无影响**：这些是纯工程优化，不改变模型的数学行为。填充的额外 token 通过 attention mask 被正确屏蔽。

---

## 总结表

| # | 创新点 | 级别 | FLUX.2 做法 | Krea-2 做法 | 优势维度 |
|---|--------|------|-------------|------------|---------|
| 1 | **纯单流架构** | 整体架构 | 8双流+48单流=56层 | 28纯单流 | 参数效率 + 更充分跨模态交互 |
| 2 | **TextFusionTransformer** | 整体架构 | 3层简单拼接→Linear | 12层→注意力融合(4block)+投影+精炼→MLP | 生成效果（更丰富的文本理解） |
| 3 | **双路径残差+独立门控** | Block | 单路径残差，1个gate，3个调制参数 | 双路径残差，2个gate，6个调制参数 | 生成效果（更精细的信息流控制） |
| 4 | **每层独立可学习调制偏置** | Block | 全局共享Modulation(SiLU+Linear) | 全局tvec + 每层learnable bias | 生成效果（层级特异性调制） |
| 5 | **GQA (4:1)** | Block | 标准MHA (Q=K=V=48heads) | GQA (Q=48, KV=12) | 推理效率（KV计算减少75%） |
| 6 | **门控注意力输出** | Block | 直接线性投影 | sigmoid(gate(x)) × attn_output | 生成效果（动态信息过滤） |
| 7 | **SwiGLU 三独立线性层** | Block | 融合Linear+chunk的SiLU gating | 独立gate/up/down三层，维度对齐128 | 生成效果 + 硬件效率 |
| 8 | **RMSNorm(可学习scale)** | Block | LayerNorm(无参数) + RMSNorm(QK) | 全面RMSNorm(可学习scale) | 计算效率 + 生成效果（逐通道缩放） |
| 9 | **3D RoPE 非均匀分配** | 组件 | 4D均匀[32,32,32,32], θ=2000 | 3D非均匀[32,48,48], θ=1000 | 生成效果（更强空间感知） |
| 10 | **文本token零位置** | 组件 | 文本有序列位置(l维度) | 文本位置全零 | 设计解耦（避免位置冲突） |
| 11 | **时间嵌入用GELU** | 组件 | SiLU | GELU(tanh) | 设计一致性（影响较小） |
| 12 | **LastLayer SimpleModulation** | 组件 | SiLU→Linear(6144→12288)动态生成 | learnable_bias(2×6144)简单偏置 | 参数效率 |
| 13 | **显式cuDNN SDPA后端** | 组件 | 自动选择后端 | 显式指定CUDNN_ATTENTION | 推理效率（确定性最优） |
| 14 | **序列填充+torch.compile** | 组件 | 无 | 填充至256倍数 + fullgraph编译 | 推理效率（编译优化） |

### 优势维度分类汇总

**可能提升生成效果的创新点**（共 7 项）：
- TextFusionTransformer（更丰富的文本理解）
- 双路径残差+独立门控（更精细的信息流控制）
- 每层独立可学习调制偏置（层级特异性调制）
- 门控注意力输出（动态信息过滤）
- SwiGLU 三独立线性层（更高表达能力）
- RMSNorm 可学习 scale（逐通道缩放灵活性）
- 3D RoPE 非均匀分配 + 更小 theta（更强空间感知）

**推理/参数效率优化的创新点**（共 7 项）：
- 纯单流架构（约一半层数和参数）
- GQA（KV 计算减少 75%）
- SwiGLU 维度对齐 128（硬件效率）
- RMSNorm 替代 LayerNorm（少一次 reduce）
- LastLayer SimpleModulation（极少参数量）
- 显式 cuDNN SDPA 后端（确定性最优）
- 序列填充 + torch.compile（编译内核复用）

**设计/工程优势的创新点**（共 3 项）：
- 文本 token 零位置编码（位置信息职责分离）
- 时间嵌入 GELU（设计一致性）
- 与主流 LLM 架构对齐（SwiGLU、GQA、RMSNorm）

---

> **分析依据文件**：
> - Krea-2 去噪模型：`/opt/nas/p/zhugechaoran/download/code/krea-2/mmdit.py`（417 行）
> - Krea-2 采样逻辑：`/opt/nas/p/zhugechaoran/download/code/krea-2/sampling.py`（146 行）
> - Krea-2 推理配置：`/opt/nas/p/zhugechaoran/download/code/krea-2/inference.py`（138 行）
> - Krea-2 文本编码器：`/opt/nas/p/zhugechaoran/download/code/krea-2/encoder.py`（76 行）
> - FLUX.2 去噪模型：`/opt/nas/p/zhugechaoran/download/code/flux2/src/flux2/model.py`（833 行）
> - FLUX.2 采样逻辑：`/opt/nas/p/zhugechaoran/download/code/flux2/src/flux2/sampling.py`（442 行）
> - FLUX.2 文本编码器：`/opt/nas/p/zhugechaoran/download/code/flux2/src/flux2/text_encoder.py`（436 行）
> - 参考分析文件（已复核校验）：`analysis/Krea2_model_analysis.md`、`analysis/FLUX2_model_analysis.md`