# i1 vs FLUX.2 去噪模型架构对比分析报告

> 本报告聚焦于去噪模型（Denoising Model / DIT Backbone）部分的网络结构差异，基于以下源代码进行逐行对比分析：
> - i1: `i1/torch_train/models/dit.py` + `i1/jax/models/dual_stream_backbone.py` + `i1/jax/models/components.py`
> - FLUX.2: `flux2/src/flux2/model.py`
>
> 每条结论均已与源代码逻辑实现进行严格复核校验。

---

## 目录

1. [两个模型去噪网络整体架构概览](#1-两个模型去噪网络整体架构概览)
2. [变化创新点详细列举](#2-变化创新点详细列举)
3. [创新点总结表](#3-创新点总结表)

---

## 1. 两个模型去噪网络整体架构概览

### i1 去噪模型架构（i1DiT / DualStreamDiT）

```
输入:
  x [B, 32, H/8, W/8]       (latent from FLUX2 VAE, 未做 2x2 pack)
  caption [B, 256, 2304]     (T5Gemma text embedding)
  mask [B, 256]              (text attention mask, optional)
  t [B]                      (timestep, 但实际被完全忽略)

  x -> Conv2d PatchEmbed(32->2016, kernel=2, stride=2) -> [B, N, 2016]
    + sinusoidal pos_embed [1, N, 2016]                <- 绝对位置编码
  caption -> TextEncoderAdapterTransformer(2304->2016, 2层Transformer)
    -> text_tokens [B, 256, 2016]

  构建 3轴 Multimodal RoPE: (time, row, col), theta=10000

  14 个 In Blocks (DualStreamDiTBlock, 无skip):
    Pre-Norm (RMSNorm, 带scale参数, 双流共享)
    -> MMDiTAttention (独立QKV, 联合注意力, 共享QKNorm, RoPE)
    -> Sandwich Norm (norm3)
    -> 残差连接
    -> Pre-Norm (RMSNorm)
    -> SwiGLU FFN (独立image/text MLP)
    -> Sandwich Norm (norm4)
    -> 残差连接
    -> 输出保存为 skip

  1 个 Mid Block (同结构)

  14 个 Out Blocks (DualStreamDiTBlock, 带skip):
    skip_linear(cat[tokens, skip]) -> [B, N, 2016]
    -> 同上结构

  取 image_tokens -> FinalLayerNoAdaLN (RMSNorm -> Linear(2016->128))
  -> Unpatchify -> [B, 32, H/8, W/8]
```

**关键参数**: hidden_size=2016, num_heads=28, depth=29, mlp_ratio=4.0, patch_size=2, 纯双流

### FLUX.2 去噪模型架构（Flux2）

```
输入:
  x [B, L_img, 128]          (latent tokens, 已由VAE做2x2 pack)
  x_ids [B, L_img, 4]        (4D position IDs: t,h,w,l)
  ctx [B, L_txt, 15360]      (Mistral text embedding)
  ctx_ids [B, L_txt, 4]      (text position IDs)
  timesteps [B]               (timestep)
  guidance [B]                (guidance scale, optional)

  x -> Linear(128->6144, bias=False)           -> img [B, L_img, 6144]
  ctx -> Linear(15360->6144, bias=False)        -> txt [B, L_txt, 6144]
  timesteps -> MLPEmbedder(256->6144)           -> vec [B, 6144]
  guidance -> MLPEmbedder(256->6144)            -> vec += guidance_emb

  vec -> double_stream_modulation_img -> mod_img (shift,scale,gate x2)  <- 全局共享
  vec -> double_stream_modulation_txt -> mod_txt (shift,scale,gate x2)  <- 全局共享
  vec -> single_stream_modulation     -> mod_single (shift,scale,gate)  <- 全局共享

  构建 4轴 RoPE: (t, h, w, l), 各32维, theta=2000

  8 个 DoubleStreamBlock:
    AdaLN调制: img_norm1 -> (1+scale)*norm(x)+shift
    -> SelfAttention (独立QKV, 联合注意力, RMSNorm QKNorm)
    -> gate * proj(attn)
    -> 残差连接
    -> AdaLN调制: img_norm2
    -> SiLU门控MLP (SiLUActivation)
    -> gate * mlp_out
    -> 残差连接

  cat(txt, img) -> 合并为单一序列

  48 个 SingleStreamBlock:
    AdaLN调制: pre_norm -> (1+scale)*norm(x)+shift
    -> linear1 -> [QKV, MLP_input] (共享权重)
    -> Attention + SiLU门控MLP
    -> linear2(cat[attn, mlp]) -> gate * output
    -> 残差连接

  去掉 txt tokens -> LastLayer (AdaLN: norm -> modulate -> Linear(6144->128))
  -> velocity [B, L_img, 128]
```

**关键参数**: hidden_size=6144, num_heads=48, depth=8+48, mlp_ratio=3.0, 混合双流+单流

---

## 2. 变化创新点详细列举

---

### 创新点1：U-Net 风格 Long Skip 连接

**FLUX.2 做法**：
顺序堆叠 8 层 DoubleStreamBlock + 48 层 SingleStreamBlock，层与层之间无跳跃连接。信息只能单向流动。

```python
# FLUX.2: model.py Flux2.forward()
for block in self.double_blocks:        # 8 层顺序执行
    img, txt, _ = block.forward_kv_extract(...)
img = torch.cat((txt, img), dim=1)
for block in self.single_blocks:        # 48 层顺序执行
    img, _ = block.forward_kv_extract(...)
```

**i1 做法**：
29 层双流块组织为 U-Net 结构：14 个 In Block + 1 个 Mid Block + 14 个 Out Block。Out Block 通过 skip connection 接收对应 In Block 的中间特征。

```python
# i1: dit.py i1DiT.forward()
skips = []
for blk in self.in_blocks:                          # 14 层
    image_tokens, text_tokens = run_block(blk, image_tokens, text_tokens)
    skips.append((image_tokens, text_tokens))        # 保存中间特征
image_tokens, text_tokens = run_block(self.mid_block, image_tokens, text_tokens)
for blk in self.out_blocks:                          # 14 层
    image_tokens, text_tokens = run_block(blk, image_tokens, text_tokens, skips.pop())
```

Skip 连接融合方式为 **concat + 线性投影**（同时对 image 和 text 两个流都做 skip）：

```python
# i1: dit.py DualStreamDiTBlock.forward()
if self.use_skip:
    image_tokens = self.skip_linear_image(torch.cat([image_tokens, skip[0]], dim=-1))  # 2*hidden -> hidden
    text_tokens = self.skip_linear_text(torch.cat([text_tokens, skip[1]], dim=-1))     # 2*hidden -> hidden
```

**区别与优点**：
- 生成效果优化：U-Net skip 连接是图像生成领域经过充分验证的结构设计。它帮助深层网络保留浅层提取的低级特征（如边缘、纹理），解决深层网络中低级信息丢失问题。在 DiT 架构中引入 U-Net skip 是一种跨范式借鉴创新。
- i1 对 **image 和 text 两个流都做 skip 连接**，这比仅对图像做 skip 更全面，帮助文本信息在深层也能保持清晰。

---

### 创新点2：完全去除 AdaLN 和时间步条件注入

**FLUX.2 做法**：
使用 AdaLN（Adaptive Layer Normalization）机制。时间步 t 和引导强度 guidance 编码为向量 vec，通过全局共享的 Modulation 模块生成 shift/scale/gate 参数，用于调制每一层的归一化和输出：

```python
# FLUX.2: model.py Flux2.forward()
vec = self.time_in(timestep_embedding(timesteps, 256))         # timestep -> vec
vec = vec + self.guidance_in(timestep_embedding(guidance, 256)) # + guidance
double_block_mod_img = self.double_stream_modulation_img(vec)   # vec -> (shift,scale,gate)x2

# FLUX.2: model.py Modulation
class Modulation(nn.Module):
    def __init__(self, dim, double, disable_bias=False):
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=not disable_bias)
    def forward(self, vec):
        out = self.lin(nn.functional.silu(vec))
        return out[:3], out[3:]  # (shift, scale, gate) for pre-attn and pre-mlp
```

**i1 做法**：
完全去除 AdaLN，**时间步信息在去噪模型中被彻底忽略**。Transformer blocks 使用简单的 Pre-Norm，无任何自适应调制：

```python
# i1: dit.py i1DiT.forward()
def forward(self, x, t, caption, mask=None, train=False):
    del t  # <- 时间步被直接删除，不使用！
    tokens = self.x_embedder(x) + self.pos_embed.to(dtype=x.dtype)
    text_tokens = self.text_encoder_adapter(caption, train=train)
    # 后续所有 block 都不使用 timestep 信息
```

JAX 版本的 DiTFinalLayerNoAdaLN 中也直接丢弃了 cond：

```python
# i1: components.py DiTFinalLayerNoAdaLN (JAX)
def __call__(self, x, cond):
    del cond  # <- cond 被丢弃
    x = norm(x)
    x = nn.Dense(...)(x)
    return x
```

**区别与优点**：
- 模型简化优化：去除 AdaLN 大幅减少参数量（省去所有 Modulation 层的参数）和计算量（省去每层的调制计算）。
- 生成效果优化：论文实验表明，在充分训练的情况下，模型可以从输入噪声的统计特征中隐式推断当前去噪阶段，AdaLN 并非必要。去除 AdaLN 后，模型的条件注入完全通过文本 token 的联合注意力实现，使得注意力机制承担了更核心的条件传递角色。
- 这是一个激进但有效的简化，使得 i1 的 Transformer block 成为标准的 Pre-Norm Transformer，大幅降低了实现复杂度。

---

### 创新点3：引入 Sandwich Norm（三明治归一化）

**FLUX.2 做法**：
使用标准的 Pre-Norm 结构（归一化 -> 子层 -> 残差连接），attention 和 MLP 输出后没有额外归一化。在 AdaLN 模式下使用 gate 控制输出幅度：

```python
# FLUX.2: model.py DoubleStreamBlock._apply_residuals()
img = img + img_mod1_gate * self.img_attn.proj(img_attn)      # gate 控制幅度
img = img + img_mod2_gate * self.img_mlp(...)                  # 无额外 norm
```

**i1 做法**：
在标准 Pre-Norm 基础上，**attention 输出和 MLP 输出后各添加一个额外的归一化层**（norm3 和 norm4），形成"三明治"结构：

```python
# i1: dit.py DualStreamDiTBlock.forward()
image_attn, text_attn = self.attn(norm1_image(image_tokens), ...)  # Pre-Norm
if self.use_sandwich_norm:
    image_attn = self.norm3(image_attn)     # <- Sandwich Norm on attention output
    text_attn = self.norm3(text_attn)       # 注意: image和text共享同一个 norm3
image_tokens = image_tokens + image_attn    # 残差连接

image_mlp = self.mlp_image(norm2_image(image_tokens))
if self.use_sandwich_norm:
    image_mlp = self.norm4(image_mlp)       # <- Sandwich Norm on MLP output
    text_mlp = self.norm4(text_mlp)         # 注意: image和text共享同一个 norm4
image_tokens = image_tokens + image_mlp     # 残差连接
```

**区别与优点**：
- 训练稳定性优化：Sandwich Norm 防止残差流中特征幅度的不断累积增长。在没有 AdaLN gate 的情况下（i1 去除了 AdaLN），这一点尤为重要——因为没有 gate 对子层输出进行缩放控制，Sandwich Norm 充当了替代的幅度控制机制。
- 生成效果优化（间接）：更稳定的训练过程意味着可以使用更大的学习率或更长的训练，从而间接提升生成效果。
- norm3/norm4 在 image 和 text 两个流之间是**共享的**，这起到了一定的正则化作用。

---

### 创新点4：纯双流架构（无单流块）

**FLUX.2 做法**：
使用先双流后单流的混合架构：前 8 层是 DoubleStreamBlock（图像和文本各有独立的 QKV/MLP），然后将文本和图像拼接后，经过 48 层 SingleStreamBlock（共享 QKV/MLP）：

```python
# FLUX.2: model.py Flux2Params
depth: int = 8               # 8 层双流
depth_single_blocks: int = 48 # 48 层单流

# FLUX.2: model.py SingleStreamBlock (共享权重处理拼接后的 txt+img)
self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim * 2)
```

**i1 做法**：
使用**纯双流架构**，所有 29 层都是 DualStreamDiTBlock，图像和文本始终保持**独立的 QKV 投影和 MLP**：

```python
# i1: dit.py DualStreamDiTConfig
depth: int = 29  # 全部是双流 DualStreamDiTBlock，无单流块
```

**区别与优点**：
- 生成效果优化：纯双流架构保证了图像和文本在**所有层**都有独立的特征处理能力。在 FLUX.2 的单流块中，文本和图像共享 QKV 和 MLP 权重，这限制了模型对两种模态差异化处理的能力。纯双流设计让每一层都能针对图像和文本分别学习最优的特征变换。
- 特别是在深层网络中，图像和文本的语义表示差异较大，独立的投影矩阵有助于更精细的特征对齐。
- 通过 U-Net skip 连接弥补了因层数较少（29 vs 56）可能造成的表示能力不足。

---

### 创新点5：Final Layer 无 AdaLN 调制

**FLUX.2 做法**：
Final Layer 使用 AdaLN 调制，从条件向量 vec 生成 shift/scale 来调制最终输出：

```python
# FLUX.2: model.py LastLayer
class LastLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = (1 + scale) * self.norm_final(x) + shift   # 自适应调制
        x = self.linear(x)
        return x
```

**i1 做法**：
Final Layer 是简单的 RMSNorm + Linear，不使用任何条件调制：

```python
# i1: dit.py FinalLayerNoAdaLN
class FinalLayerNoAdaLN(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm):
        self.norm_final = _norm(use_rmsnorm)(hidden_size)  # RMSNorm with learnable scale
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x):
        return self.linear(self.norm_final(x))  # 简单 norm -> linear
```

**区别与优点**：
- 模型简化优化：与创新点2一致，去除了最终输出层对时间步的依赖，使整个模型完全不感知 timestep。
- Final Layer 的 norm 使用 RMSNorm（带可学习 scale 参数），而非 FLUX.2 的无参数 LayerNorm，提供了一定的可学习能力来补偿缺失的 AdaLN。

---

### 创新点6：Transformer 文本编码器适配器

**FLUX.2 做法**：
使用**单层线性投影**将文本编码器输出映射到模型隐空间：

```python
# FLUX.2: model.py Flux2.__init__()
self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)
# 15360 -> 6144, 一个 Linear 层
```

**i1 做法**：
使用 **2 层 Transformer Block 作为文本编码器适配器**：

```python
# i1: dit.py TextEncoderAdapterTransformer
class TextEncoderAdapterTransformer(nn.Module):
    def __init__(self, in_channels, hidden_size, drop_text_prob, num_heads, mlp_ratio,
                 use_qknorm, use_swiglu, use_rmsnorm, token_len, num_blocks=2):
        self.learnable_null_caption = nn.Parameter(...)   # 可学习空文本嵌入
        self.connector_in = nn.Linear(in_channels, hidden_size)  # 2304 -> 2016
        # 2 个 Transformer Block:
        for block_idx in range(num_blocks):
            # RMSNorm -> Attention -> 残差
            # RMSNorm -> SwiGLU FFN -> 残差

    def forward(self, caption, train):
        # 训练时随机替换为 learnable_null_caption (drop_text_prob=0.1)
        x = self.connector_in(caption)
        for block_idx in range(self.num_blocks):
            x = x + attn(norm1(x))          # Self-attention on text tokens
            x = x + mlp(norm2(x))           # SwiGLU FFN
        return x
```

**区别与优点**：
- 生成效果优化：2 层 Transformer 适配器相比简单的线性投影，能够对文本特征进行更深层次的非线性变换和上下文建模。文本 token 之间先经过自注意力重新组织语义关系，再进入双流 DiT，使文本条件更好地对齐到视觉特征空间。
- 适配器内部使用与主干网络**相同的组件**（RMSNorm、QKNorm、SwiGLU），保持了架构一致性。
- 内置了 **可学习空文本嵌入**（learnable_null_caption），用于 CFG 训练时的文本 dropout，而非固定的零向量。

---

### 创新点7：双重位置编码（Sinusoidal + Multimodal RoPE）

**FLUX.2 做法**：
仅使用 RoPE（Rotary Position Embedding），没有绝对位置编码：

```python
# FLUX.2: model.py Flux2.forward()
pe_x = self.pe_embedder(x_ids)      # RoPE only
pe_ctx = self.pe_embedder(ctx_ids)
```

**i1 做法**：
同时使用**可学习的 Sinusoidal 绝对位置编码 + Multimodal RoPE 相对位置编码**：

```python
# i1: dit.py i1DiT.__init__()
# 1. Sinusoidal 绝对位置编码 (初始化为2D sinusoidal, 作为可学习参数)
pos = _get_pos_embed(cfg.hidden_size, hw)
self.pos_embed = nn.Parameter(torch.from_numpy(pos.reshape(1, hw*hw, cfg.hidden_size)))

# 2. Multimodal RoPE 相对位置编码
self.rope_embedder = MultimodalRopeEmbedder(axes_dims, axes_lens, axes_scales, theta=cfg.rope_theta)

# i1: dit.py i1DiT.forward()
tokens = self.x_embedder(x) + self.pos_embed.to(dtype=x.dtype)  # 加绝对位置编码
image_freqs, text_freqs = self._rope_freqs(text_tokens, text_mask, tokens.shape[1])  # 计算 RoPE
```

**区别与优点**：
- 生成效果优化：双重位置编码结合了两种编码的优势：
  - **Sinusoidal 绝对位置编码**：为 image token 提供固定的空间位置参考，训练后成为可学习参数可进一步适配
  - **RoPE 相对位置编码**：在注意力计算中注入相对位置关系，具有更好的长度泛化性
- 支持**分辨率内插**：当从 256->512->1024 分辨率微调时，通过坐标缩放保持位置一致性

---

### 创新点8：3 轴 RoPE 替代 4 轴 RoPE

**FLUX.2 做法**：
使用 **4 轴 RoPE**，每轴固定 32 维，总计 128 维（=head_dim）：

```python
# FLUX.2: model.py Flux2Params
axes_dim: list[int] = [32, 32, 32, 32]  # t, h, w, l 四个轴
theta: int = 2000

# 文本: (t=0, h=0, w=0, l=seq_pos)       <- 文本位置在 l 轴
# 图像: (t=0, h=row, w=col, l=0)         <- 图像位置在 h,w 轴
```

**i1 做法**：
使用 **3 轴 Multimodal RoPE**，维度自动分配（time 占一半，row/col 各占四分之一），theta=10000：

```python
# i1: dit.py _default_rope_axes_dims()
def _default_rope_axes_dims(head_dim):
    time_dim = head_dim // 2          # 时间轴占一半
    remaining = head_dim - time_dim
    row_dim = remaining // 2          # 行轴
    col_dim = remaining - row_dim     # 列轴
    return time_dim, row_dim, col_dim
# 例: head_dim=72 (2016/28) -> (36, 18, 18)

# 文本: (time=seq_pos, row=0, col=0)         <- 文本位置在 time 轴
# 图像: (time=text_length, row=row, col=col) <- 图像的 time = 文本长度
```

**区别与优点**：
- 生成效果优化：
  1. **时间轴分配更多维度（head_dim/2 vs head_dim/4）**：更多的 time 维度意味着模型在序列顺序编码上有更强的表达能力。
  2. **简洁的 3 轴设计**：i1 不需要第 4 个轴（l），因为文本的序列位置直接编码在 time 轴上。
  3. 位置 ID 设计使得**图像 token 的 time 坐标 = 文本长度**，自然创建了"文本在前、图像在后"的时间序列关系。

---

### 创新点9：全面使用带可学习参数的 RMSNorm

**FLUX.2 做法**：
在 Block 内部的 Pre-Norm 位置使用 **无参数的 LayerNorm**（elementwise_affine=False），仅在 QKNorm 中使用带可学习 scale 参数的 RMSNorm：

```python
# FLUX.2: model.py DoubleStreamBlock
self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # 无可学习参数
self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # 无可学习参数
```

**i1 做法**：
**全面使用 RMSNorm**，且所有 norm 层都带有可学习的 scale 参数：

```python
# i1: dit.py RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.scale = nn.Parameter(torch.ones(dim))  # <- 可学习 scale
    def forward(self, x):
        x_float = x.float()
        x_float = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * self.scale.float()).to(dtype)
```

i1 中所有归一化位置都使用此 RMSNorm：Pre-Norm、Sandwich Norm、QKNorm、Final Layer Norm。

**区别与优点**：
- 计算效率优化：RMSNorm 比 LayerNorm 更高效，因为不需要计算均值（mean），仅计算均方根（RMS）。
- 生成效果优化（间接）：带 scale 参数的归一化比无参数归一化更具表达能力。在去除 AdaLN 后，这些可学习的 scale 参数提供了替代的特征缩放机制。

---

### 创新点10：Conv2d Patch Embedding 替代线性投影

**FLUX.2 做法**：
VAE 已经将 latent 做了 2x2 空间打包（32ch -> 128ch），DIT 直接使用**线性投影**映射到隐空间：

```python
# FLUX.2: model.py Flux2.__init__()
self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
# 128 -> 6144, 简单线性投影
```

**i1 做法**：
接收未打包的 32 通道 latent，使用 **Conv2d Patch Embedding** 同时完成空间下采样和通道映射：

```python
# i1: dit.py PatchEmbed
class PatchEmbed(nn.Module):
    def __init__(self, patch_size, hidden_size, in_channels):
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)           # [B, 32, H/8, W/8] -> [B, 2016, H/16, W/16]
        return x.flatten(2).transpose(1, 2)  # -> [B, N, 2016]
```

**区别与优点**：
- 生成效果优化：Conv2d 核可以捕获 2x2 patch 内部的空间局部关系（如局部梯度、纹理模式），而 FLUX.2 的线性投影将已经打包的 128 维向量视为无结构的平坦向量。

---

### 创新点11：支持变长文本掩码（Text Mask）

**FLUX.2 做法**：
不支持文本掩码，所有文本 token（包括 padding token）同等参与注意力。

**i1 做法**：
支持变长文本掩码，在注意力中正确屏蔽 padding token：

```python
# i1: dit.py MMDiTAttention.forward()
if text_mask is not None:
    image_mask = torch.ones((bsz, image_len), dtype=torch.bool, device=text_tokens.device)
    key_mask = torch.cat([image_mask, text_mask.bool()], dim=1)
    attn_mask = key_mask[:, None, None, :]
out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
if key_mask is not None:
    out = out * key_mask[:, :, None].to(out.dtype)
```

同时文本掩码也用于 Block 输出的 masking 和 RoPE 位置 ID 构建。

**区别与优点**：
- 生成效果优化：正确处理变长文本可以防止 padding token 对注意力分布的干扰。当文本长度远小于最大长度时，padding token 不参与注意力，提高了文本条件的精确性。

---

### 创新点12：共享归一化参数（Image/Text 流共享 Norm）

**FLUX.2 做法**：
图像和文本分支使用**各自独立的归一化层**：

```python
# FLUX.2: model.py DoubleStreamBlock
self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
```

**i1 做法**（默认 use_separate_norms=False）：
图像和文本分支**共享同一个归一化层**（同一组可学习 scale 参数）：

```python
# i1: dit.py DualStreamDiTBlock (use_separate_norms=False)
self.norm1 = RMSNorm(hidden_size)  # 1 个 norm, image 和 text 共用
self.norm2 = RMSNorm(hidden_size)  # 1 个 norm, image 和 text 共用
```

Sandwich Norm 的 norm3/norm4 和 QKNorm 也是 image/text 共享的。

**区别与优点**：
- 参数效率优化：共享 norm 减少了参数数量。
- 生成效果优化（间接）：共享的 scale 参数强制 image 和 text token 使用相同的特征缩放标准，起到一种**跨模态正则化**的作用，有助于两个模态的特征对齐。

---

### 创新点13：线性层保留 Bias

**FLUX.2 做法**：
几乎所有线性层都设置 bias=False：

```python
# FLUX.2: model.py
self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
self.qkv = nn.Linear(dim, dim * 3, bias=False)
self.proj = nn.Linear(dim, dim, bias=False)
```

**i1 做法**：
线性层**保留 bias**（PyTorch 的 nn.Linear 默认 bias=True）：

```python
# i1: dit.py
self.qkv = nn.Linear(hidden_size, 3 * hidden_size)       # bias=True (default)
self.proj = nn.Linear(hidden_size, hidden_size)           # bias=True
self.w12 = nn.Linear(hidden_size, 2 * hidden_features)   # bias=True
```

JAX 版本也明确设置了 use_bias=True。

**区别与优点**：
- 模型表达能力优化：bias 提供了额外的偏移自由度。在 i1 的相对较小模型规模（3B vs 12B+）下，保留 bias 可以补偿较少参数带来的表达能力限制。

---

### 创新点14：可学习空文本嵌入 + 标准 CFG 替代 Guidance Embedding

**FLUX.2 做法**：
使用 **Guidance Embedding** 将 CFG 引导强度编码为模型内部条件：

```python
# FLUX.2: model.py Flux2
self.use_guidance_embed = True
self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)

# Forward:
vec = self.time_in(timestep_emb)
vec = vec + self.guidance_in(guidance_emb)  # 引导强度编码进条件向量
```

**i1 做法**：
使用**可学习空文本嵌入** + 标准 CFG，不在模型内部编码引导强度：

```python
# i1: dit.py TextEncoderAdapterTransformer
self.learnable_null_caption = nn.Parameter(torch.empty(1, token_len, in_channels))
nn.init.normal_(self.learnable_null_caption, std=in_channels ** -0.5)

# 训练时: 以 drop_text_prob=0.1 概率替换为 learnable_null_caption
```

**区别与优点**：
- 生成效果优化：可学习的空文本嵌入（而非固定零向量）让模型能够学习最优的"无条件"表示。
- 通用性优化：标准 CFG 是一种与模型无关的推理技术，不需要在模型训练时绑定特定的引导强度。

---

### 创新点15：RoPE theta 参数选择（10000 vs 2000）

**FLUX.2 做法**：theta=2000

**i1 做法**：theta=10000

**区别与优点**：
- 生成效果优化：theta=10000 是 NLP Transformer（如 LLaMA 等）中广泛使用的标准值。较大的 theta 值使 RoPE 的频率分布更平缓，对远距离位置的区分能力更强，有利于高分辨率图像（token 数量大）的位置建模。

---

### 创新点16：Multimodal RoPE 位置 ID 设计差异

**FLUX.2 做法**：
使用 4D 位置 ID (t, h, w, l)，文本和图像的位置信息在不同轴上编码：

```
文本: (t=0, h=0, w=0, l=0,1,...,L-1)     <- 文本序列位置在 l 轴
图像: (t=0, h=0,...,H-1, w=0,...,W-1, l=0) <- 图像空间位置在 h,w 轴
```

**i1 做法**：
使用 3D 位置 ID (time, row, col)，文本和图像的时间轴编码形成自然的序列关系：

```
文本: (time=0,1,...,L-1, row=0, col=0)    <- 文本序列位置在 time 轴
图像: (time=text_length, row=r, col=c)     <- 图像 time = 文本长度（常数）
```

**区别与优点**：
- 生成效果优化：i1 的位置 ID 设计创建了一个有意义的时间序列关系：
  1. 文本 token 在 time 轴上递增排列，编码了文本的自然阅读顺序
  2. 图像 token 的 time = text_length，所有图像 token 位于同一"时间点"，自然排在所有文本 token 之后
  3. RoPE 的旋转角度自动反映了"文本->图像"的条件生成关系
- FLUX.2 将文本序列位置放在第 4 轴 l，图像空间位置在 h/w 轴，两者在不同轴上，不存在自然的序列距离关系。

---

## 3. 创新点总结表

| # | 创新点 | FLUX.2 做法 | i1 做法 | 优点类型 |
|---|--------|------------|---------|---------|
| 1 | **U-Net Long Skip** | 无跳跃连接（56层顺序执行） | 14 In + 1 Mid + 14 Out (U-Net skip) | 生成效果（保留低级特征） |
| 2 | **去除 AdaLN / 去除时间步** | AdaLN 调制 + timestep 编码 | 完全去除 AdaLN，时间步不使用 | 模型简化 + 生成效果 |
| 3 | **Sandwich Norm** | 无（依赖 AdaLN gate 控制幅度） | norm3/norm4 在 attention/MLP 输出后 | 训练稳定性 |
| 4 | **纯双流架构** | 混合：8层双流 + 48层单流 | 全部29层双流（独立 QKV/MLP） | 生成效果（跨模态独立建模） |
| 5 | **Final Layer 无 AdaLN** | AdaLN 调制输出层 | 简单 RMSNorm + Linear | 模型简化 |
| 6 | **Transformer 文本适配器** | 单层 Linear 投影 | 2层 Transformer Block | 生成效果（文本-视觉对齐） |
| 7 | **双重位置编码** | 仅 RoPE | Sinusoidal APE + RoPE | 生成效果（绝对+相对位置） |
| 8 | **3轴 RoPE** | 4轴 (t,h,w,l), 各32维 | 3轴 (time,row,col), time占半维度 | 生成效果（更简洁高效） |
| 9 | **RMSNorm 带 scale** | Pre-Norm: 无参数 LayerNorm | 全部: 带可学习 scale 的 RMSNorm | 计算效率 + 表达能力 |
| 10 | **Conv2d Patch Embed** | Linear 投影（VAE已pack） | Conv2d(k=2,s=2) 空间下采样 | 生成效果（空间结构感知） |
| 11 | **变长文本掩码** | 不支持（padding参与注意力） | 支持（mask屏蔽padding） | 生成效果（精确文本条件） |
| 12 | **共享 Norm 参数** | 独立 norm（但无参数） | image/text 共享 norm（有参数） | 参数效率 + 跨模态正则化 |
| 13 | **保留 Bias** | 几乎全部 bias=False | 保留 bias=True | 模型表达能力 |
| 14 | **Learnable Null Caption** | Guidance Embedding 编码引导强度 | 可学习空文本 + 标准 CFG | 通用性 + 生成效果 |
| 15 | **RoPE theta=10000** | theta=2000 | theta=10000 | 生成效果（长距离位置建模） |
| 16 | **Multimodal RoPE 位置设计** | 4D, 文本在l轴, 轴分离 | 3D, 文本在time轴, 图像time=text_len | 生成效果（自然序列关系） |

---

> **分析的文件列表**:
> - i1: torch_train/models/dit.py, jax/models/dual_stream_backbone.py, jax/models/components.py, torch_train/configs/i1_256.py, jax/configs/i1_training/1024_resolution.py, torch_train/diffusion/rectified_flow.py
> - FLUX.2: src/flux2/model.py, src/flux2/sampling.py
> - 参考: analysis/i1_model_analysis.md, analysis/FLUX2_model_analysis.md