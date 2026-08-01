# ERNIE-Image vs FLUX2 去噪模型架构对比分析

> 本报告基于对两个模型去噪网络部分的源代码逐行对比分析。
>
> - **ERNIE-Image 去噪模型代码**：`/opt/nas/p/conda/envs/pytorch2.5.1_zhugechaoran/lib/python3.12/site-packages/diffusers/models/transformers/transformer_ernie_image.py`（`ErnieImageTransformer2DModel`）
> - **FLUX2 去噪模型代码**：`/opt/nas/p/zhugechaoran/download/code/flux2/src/flux2/model.py`（`Flux2` 类）
> - **ERNIE-Image Pipeline 代码**：`/opt/nas/p/conda/envs/pytorch2.5.1_zhugechaoran/lib/python3.12/site-packages/diffusers/pipelines/ernie_image/pipeline_ernie_image.py`
> - **FLUX2 Sampling 代码**：`/opt/nas/p/zhugechaoran/download/code/flux2/src/flux2/sampling.py`

---

## 一、整体架构层面的变化创新点

### 创新点1：纯单流(Single-Stream)架构 vs 双流+单流混合架构

**FLUX2 的做法**：
- 采用 **双流(DoubleStreamBlock) × 8层 + 单流(SingleStreamBlock) × 48层** 的混合架构
- 双流块中，图像和文本各有**独立的 LayerNorm、Attention QKV投影、独立的MLP**，仅在Attention计算时将Q/K/V拼接进行联合注意力，然后分开做残差和MLP
- 单流块中，图像和文本token拼接后用同一组参数处理

```python
# FLUX2 model.py - 双流块中图像/文本独立参数
class DoubleStreamBlock(nn.Module):
    self.img_norm1 = nn.LayerNorm(hidden_size, ...)
    self.img_attn = SelfAttention(dim=hidden_size, ...)
    self.img_norm2 = nn.LayerNorm(hidden_size, ...)
    self.img_mlp = nn.Sequential(Linear → SiLUActivation → Linear)
    self.txt_norm1 = nn.LayerNorm(hidden_size, ...)
    self.txt_attn = SelfAttention(dim=hidden_size, ...)
    self.txt_norm2 = nn.LayerNorm(hidden_size, ...)
    self.txt_mlp = nn.Sequential(Linear → SiLUActivation → Linear)

# FLUX2 forward - 先双流后单流
for block in self.double_blocks:
    img, txt, _ = block.forward_kv_extract(img, txt, ...)
img = torch.cat((txt, img), dim=1)  # 拼接后进入单流
for block in self.single_blocks:
    img, _ = block.forward_kv_extract(img, pe, ...)
```

**ERNIE-Image 的做法**：
- 采用 **纯单流(ErnieImageSharedAdaLNBlock) × 24层** 的架构，**没有任何双流块**
- 图像token和文本token在进入Transformer层**之前**就被拼接：`x = torch.cat([img_sbh, text_sbh], dim=0)`
- 所有层使用完全相同的结构，图像和文本共享同一套Norm、Attention和MLP参数

```python
# ERNIE-Image transformer_ernie_image.py
img_sbh = self.x_embedder(hidden_states).transpose(0, 1).contiguous()
text_sbh = text_bth.transpose(0, 1).contiguous()
x = torch.cat([img_sbh, text_sbh], dim=0)  # 一开始就拼接
for layer in self.layers:  # 24层完全相同的单流块
    x = layer(x, rotary_pos_emb, temb, attention_mask)
```

**是否为优点及分析**：
- **架构简洁性优点**：纯单流架构更加简洁统一，减少了工程复杂度和代码维护成本
- **参数效率优点**：在双流块中，图像和文本各有独立的Norm、QKV投影和MLP，参数量是单流的约2倍（每层）。纯单流设计让每层参数更精简
- **图文深度交互优点**：从第一层开始图像和文本token就在同一个注意力空间中交互，而FLUX2的前8层双流块中图文仅通过注意力的QKV拼接做有限交互。更早的深度融合有助于更好的图文对齐
- **潜在的生成效果影响**：双流设计的优势在于早期给图像和文本各自独立的表示空间，避免早期交互过深导致特征混淆。纯单流舍弃了这种设计可能在需要高度精确文本语义保留的场景下略有不同。但ERNIE-Image实际生成效果表明纯单流架构是可行且有效的

---

### 创新点2：共享AdaLN调制参数——单一全局调制模块

**FLUX2 的做法**：
- 使用 **3个独立的全局Modulation模块**：
  - `double_stream_modulation_img`：为所有双流块的图像分支生成调制参数（shift/scale/gate × 2 组）
  - `double_stream_modulation_txt`：为所有双流块的文本分支生成调制参数（shift/scale/gate × 2 组）
  - `single_stream_modulation`：为所有单流块生成调制参数（shift/scale/gate × 1 组）
- 每个Modulation产生不同数量的参数：双流的产生6个参数（2组 × 3），单流的产生3个参数

```python
# FLUX2 model.py
self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, disable_bias=True)
self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True, disable_bias=True)
self.single_stream_modulation = Modulation(self.hidden_size, double=False, disable_bias=True)

# Modulation类输出: SiLU → Linear → chunk成3或6个参数
class Modulation(nn.Module):
    def forward(self, vec):
        out = self.lin(nn.functional.silu(vec))
        out = out.chunk(self.multiplier, dim=-1)  # multiplier=6(双流) or 3(单流)
        return out[:3], out[3:] if self.is_double else None
```

**ERNIE-Image 的做法**：
- 使用 **单一的 `adaLN_modulation` 模块**，产生6个参数（shift/scale/gate × 2 组），在**所有24层和所有token类型（图像+文本）之间完全共享**
- 调制参数在序列维度上广播到每个token位置

```python
# ERNIE-Image transformer_ernie_image.py
self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
nn.init.zeros_(self.adaLN_modulation[-1].weight)  # 零初始化
nn.init.zeros_(self.adaLN_modulation[-1].bias)     # 零初始化

# forward中：
c = self.time_embedding(sample)
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
    t.unsqueeze(0).expand(S, -1, -1).contiguous() 
    for t in self.adaLN_modulation(c).chunk(6, dim=-1)
]
```

**是否为优点及分析**：
- **参数效率优点**：FLUX2的3个Modulation模块（每个都是 `Linear(hidden_size, multiplier * hidden_size)`）比ERNIE-Image的单一模块使用了更多参数。ERNIE-Image用1个模块取代了3个，显著减少了Modulation相关的参数量
- **零初始化设计优点**：ERNIE-Image对adaLN_modulation的权重和bias都做了**零初始化**，这意味着训练初期调制参数近似为0（shift≈0, scale≈0, gate≈0），Transformer块的行为接近恒等映射。这有助于训练初期的稳定性。FLUX2的Modulation没有明确的零初始化
- **简洁性优点**：单一模块设计在概念上更简洁，不需要区分不同流的调制策略
- **潜在取舍**：FLUX2为图像和文本分支提供独立的调制参数，理论上可以让不同模态获得不同的时间步条件化行为。ERNIE-Image让所有token共享同一组调制参数，减少了这种灵活性，但简化了模型结构

---

### 创新点3：更紧凑的模型层数设计

**FLUX2 的做法**：
- 总层数 = 8（双流）+ 48（单流）= **56层**
- hidden_size = 6144, num_heads = 48

**ERNIE-Image 的做法**：
- 总层数 = **24层**（纯单流）
- hidden_size = 3072, num_heads = 24

**是否为优点及分析**：
- **推理效率优点**：24层比56层少了约57%的层数，hidden_size减半（3072 vs 6144），理论上推理速度显著更快
- **这并非仅仅是生成效果优化**，而是**推理效率和模型规模的优化**。以更小的模型规模实现有竞争力的生成质量，是ERNIE-Image的一个设计目标
- 注意：虽然ERNIE-Image声称8B参数，但hidden_size=3072、24层的配置实际参数量远小于FLUX2的32B版本。模型的有效参数量和生成效果需要在同等数据和训练策略下评估

---

## 二、单独组件层面的变化创新点

### 创新点4：GeGLU FFN vs SwiGLU FFN

**FLUX2 的做法**：
- 使用 **SwiGLU (SiLU Gated) 激活**：将输入分成两半，一半过SiLU激活函数，然后与另一半逐元素相乘
- 双流块MLP：`Linear(hidden_size → mlp_hidden_dim*2, bias=False) → SiLUActivation → Linear(mlp_hidden_dim → hidden_size, bias=False)`
- 单流块中Attention和MLP并行计算：`linear1` 同时产出QKV和MLP双份输入，最终 `linear2` 合并输出

```python
# FLUX2 model.py
class SiLUActivation(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2  # SiLU(x1) * x2 = SwiGLU
```

**ERNIE-Image 的做法**：
- 使用 **GeGLU (GELU Gated) 激活**：gate_proj和up_proj是**分离的独立Linear层**，gate过GELU激活后与up逐元素相乘
- FFN结构：`gate_proj(hidden_size → ffn_hidden_size)` + `up_proj(hidden_size → ffn_hidden_size)` → GELU(gate) * up → `linear_fc2(ffn_hidden_size → hidden_size)`

```python
# ERNIE-Image transformer_ernie_image.py
class ErnieImageFeedForward(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size):
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.linear_fc2(self.up_proj(x) * F.gelu(self.gate_proj(x)))
```

**是否为优点及分析**：
- **独立投影层的灵活性优点**：ERNIE-Image使用**分离的 gate_proj 和 up_proj**（两个独立的Linear），而FLUX2将gate和up融合在一个Linear中然后chunk分开。分离的投影层允许gate和up学习不同的投影矩阵，可能提供更好的表达能力
- **GELU vs SiLU的差异**：GELU和SiLU（Swish）是非常相似的激活函数，两者在实践中差异不大。GELU在某些NLP任务中表现略好，SiLU在某些视觉任务中表现略好，但差异通常很小
- **对生成效果的影响**：这个改变主要是FFN内部的实现细节，对最终生成效果的影响较小，更多是工程设计偏好的差异

---

### 创新点5：3维RoPE位置编码 vs 4维RoPE位置编码

**FLUX2 的做法**：
- 使用 **4维RoPE**，维度分配 `axes_dim = [32, 32, 32, 32]`（均匀分配），4个维度分别对应 t（时间）、h（高度）、w（宽度）、l（序列位置）
- `theta = 2000`
- 图像token的IDs格式：`(t, h, w, l)`，其中 l=0（图像不使用l维度），t用于区分参考图和生成图
- 文本token的IDs格式：`(t, h, w, l)`，其中 h=0, w=0, l=序列位置
- RoPE计算使用旋转矩阵形式：`[cos, -sin, sin, cos]` → 2×2矩阵

```python
# FLUX2 model.py
axes_dim: list[int] = [32, 32, 32, 32]  # 4D
theta: int = 2000

def rope(pos, dim, theta):
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)  # 旋转矩阵形式
    return out

def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)  # 矩阵乘法形式
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
```

**ERNIE-Image 的做法**：
- 使用 **3维RoPE**，维度分配 `rope_axes_dim = (32, 48, 48)`（**非均匀分配**），3个维度对应"文本序列位置/时间"、"高度"、"宽度"
- `theta = 256`（比FLUX2小得多）
- 空间维度分配更多：h和w各48维 vs FLUX2的各32维，总共96维给空间 vs 64维
- RoPE计算使用 **rotate_half** 形式（非旋转矩阵形式）

```python
# ERNIE-Image transformer_ernie_image.py
rope_axes_dim: Tuple[int, int, int] = (32, 48, 48)  # 3D
rope_theta: int = 256

class ErnieImageEmbedND3(nn.Module):
    def forward(self, ids):
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)], dim=-1)
        emb = emb.unsqueeze(2)
        return torch.stack([emb, emb], dim=-1).reshape(*emb.shape[:-1], -1)  # 复制角度用于rotate_half

# rotate_half 方式（在AttnProcessor中）
def apply_rotary_emb(x_in, freqs_cis):
    cos_ = torch.cos(freqs_cis)
    sin_ = torch.sin(freqs_cis)
    x1, x2 = x.chunk(2, dim=-1)
    x_rotated = torch.cat((-x2, x1), dim=-1)  # rotate_half: [-x2, x1]
    return x * cos_ + x_rotated * sin_
```

**是否为优点及分析**：
- **更多空间维度分配优点（可能提升生成效果）**：ERNIE-Image将head_dim=128中的96维（75%）分配给空间坐标（h=48, w=48），仅32维分配给序列/时间位置。而FLUX2均匀分配每维32维。对于图像生成任务，空间位置信息比序列位置信息更重要，更多维度给空间可以让模型更精确地感知像素的空间关系，可能**提升图像的空间一致性和结构准确性**
- **更小的theta值**：`theta=256` vs `theta=2000`。更小的theta意味着频率更高，RoPE衰减更快，模型对**近距离位置差异更敏感**。这可能有助于捕捉图像中的局部细节和纹理，但可能降低对非常远距离依赖的建模能力
- **3D vs 4D**：ERNIE-Image不使用第4维（l），因为它是纯文生图模型不需要区分参考图。而FLUX2的第4维l主要用于文本token的序列位置编码。ERNIE-Image将文本的序列位置放在第1维（与时间共用），在纯文生图场景下是合理的简化
- **整体上这是一个生成效果相关的优化**：更合理的空间维度分配对图像生成质量可能有正面影响

---

### 创新点6：Conv2d Patch嵌入 vs Linear Patch嵌入

**FLUX2 的做法**：
- 使用 **`nn.Linear`** 将已经flatten好的latent token投影到隐藏空间
- Patchify在模型外部完成（在 `sampling.py` 的 `prc_img` 函数中通过 `rearrange(x, "c h w -> (h w) c")` 将空间flatten到序列维度）

```python
# FLUX2 model.py
self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)  # 128 → 6144
# 输入 x 已经是 [B, L, 128] 的flatten形式
img = self.img_in(x)
```

**ERNIE-Image 的做法**：
- 使用 **`nn.Conv2d`** 作为patch嵌入层，直接接收 `[B, C, H, W]` 格式的latent输入
- Patchify在模型内部通过Conv2d完成（kernel_size=stride=patch_size）
- 模型接收的是原始2D spatial格式的latent，而非预处理后的1D序列

```python
# ERNIE-Image transformer_ernie_image.py
class ErnieImagePatchEmbedDynamic(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W] → [B, embed_dim, H', W']
        batch_size, dim, height, width = x.shape
        return x.reshape(batch_size, dim, height * width).transpose(1, 2).contiguous()  # → [B, H'W', embed_dim]

# 在 ErnieImageTransformer2DModel 中：
# patch_size=1, in_channels=128, hidden_size=3072
self.x_embedder = ErnieImagePatchEmbedDynamic(in_channels, hidden_size, patch_size)
# 输入 hidden_states 是 [B, 128, H, W]
img_sbh = self.x_embedder(hidden_states).transpose(0, 1).contiguous()
```

**是否为优点及分析**：
- **端到端封装优点**：Conv2d嵌入将patchify和投影合并为一步，模型接口更清晰——直接输入2D latent，无需外部预处理。而FLUX2需要在模型外部做flatten和位置ID生成
- **Conv2d的局部感知能力**：虽然当前配置 `patch_size=1` 时Conv2d等价于1×1卷积（与逐像素Linear等价），但Conv2d框架天然支持更大的patch_size，当patch_size>1时Conv2d可以捕获局部空间关系。这为未来调整patch大小提供了灵活性
- **带bias设计**：ERNIE-Image的Conv2d patch嵌入**带有bias**（`bias=True`），而FLUX2的Linear投影**不带bias**（`bias=False`）。带bias可以提供额外的偏移自由度
- **对生成效果的直接影响较小**，更多是接口设计和工程封装的优化

---

### 创新点7：RMSNorm vs LayerNorm 用于Transformer块内部归一化

**FLUX2 的做法**：
- 双流块和单流块的pre-norm使用 **`nn.LayerNorm(elementwise_affine=False)`**（无可学习参数的标准LayerNorm）
- QK Norm使用自定义 **`RMSNorm`**（带可学习scale参数）

```python
# FLUX2 model.py - SingleStreamBlock
self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

# FLUX2 model.py - DoubleStreamBlock
self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

# QKNorm
class QKNorm(torch.nn.Module):
    self.query_norm = RMSNorm(dim)
    self.key_norm = RMSNorm(dim)
```

**ERNIE-Image 的做法**：
- Transformer块的pre-norm使用 **`RMSNorm`**（从diffusers导入的RMSNorm，带可学习的elementwise_affine参数）
- QK Norm也使用 **`RMSNorm`**（通过 `nn.RMSNorm` 实现）

```python
# ERNIE-Image transformer_ernie_image.py - ErnieImageSharedAdaLNBlock
self.adaLN_sa_ln = RMSNorm(hidden_size, eps=eps)  # diffusers.models.normalization.RMSNorm
self.adaLN_mlp_ln = RMSNorm(hidden_size, eps=eps)

# QK Norm
self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)
self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)
```

**是否为优点及分析**：
- **计算效率优点**：RMSNorm相比LayerNorm省去了均值的计算和减去均值的操作，计算量更小（约省30%的norm计算）。对于大模型来说，这种效率提升是有意义的
- **表达能力**：ERNIE-Image的RMSNorm**带有可学习参数**（elementwise_affine=True），而FLUX2的LayerNorm**不带可学习参数**（elementwise_affine=False）。带可学习参数的Norm层可以为每个维度学习独立的缩放因子，提供更好的表达能力
- **训练稳定性**：RMSNorm在大型语言模型中已被广泛验证具有良好的训练稳定性（LLaMA/Qwen等大模型均使用RMSNorm）
- **对生成效果的影响**：RMSNorm与LayerNorm在实践中通常效果接近，但计算效率的提升使得在相同计算预算下可以训练更多步或使用更大的batch size，间接提升生成效果

---

### 创新点8：QKV分离投影 vs QKV融合投影

**FLUX2 的做法**：
- Attention的Q/K/V通过**单一融合的Linear层**同时产生，然后通过 `rearrange` 拆分

```python
# FLUX2 model.py
class SelfAttention(nn.Module):
    self.qkv = nn.Linear(dim, dim * 3, bias=False)  # 一个Linear同时产生Q/K/V

# DoubleStreamBlock中：
img_qkv = self.img_attn.qkv(img_modulated)
img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

# SingleStreamBlock中更进一步——linear1同时产生QKV和MLP输入：
self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim * 2, bias=False)
qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim * 2], dim=-1)
```

**ERNIE-Image 的做法**：
- Q/K/V通过 **3个独立的Linear层** 分别计算

```python
# ERNIE-Image transformer_ernie_image.py
class ErnieImageAttention(nn.Module):
    self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)  # bias=False
    self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
    self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

# AttnProcessor中：
query = attn.to_q(hidden_states)
key = attn.to_k(hidden_states)
value = attn.to_v(hidden_states)
```

**是否为优点及分析**：
- **灵活性优点**：分离的Q/K/V投影层允许独立调整Q/K/V的参数，例如可以单独对某个投影做LoRA微调或量化
- **兼容性优点**：分离的Q/K/V投影与diffusers框架的Attention接口更兼容，便于使用不同的AttnProcessor、Peft Adapter等
- **FLUX2的融合优势在于计算效率**：融合的Linear层在GPU上可以减少kernel launch次数，理论上更快。特别是SingleStreamBlock将QKV和MLP输入合并到一个Linear中，进一步减少计算开销
- **对生成效果的影响**：数学上等价（因为融合或分离的Linear对同样输入产生同样输出），对生成效果无直接影响。这是一个**工程灵活性 vs 计算效率的取舍**

---

### 创新点9：Attention输出投影带独立Linear vs 融合到Attention内部

**FLUX2 的做法**：
- 双流块：Attention输出通过 `self.img_attn.proj(img_attn)` 投影后直接加上gate调制参数做残差
- 单流块：Attention输出和MLP输出**cat到一起**后通过一个 `linear2` 统一投影

```python
# FLUX2 SingleStreamBlock
def _out(self, x, attn, mlp, mod_gate):
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))  # Attn+MLP合并后统一投影
    return x + mod_gate * output
```

**ERNIE-Image 的做法**：
- Attention有独立的输出投影 `to_out[0]`（`nn.Linear(inner_dim, out_dim, bias=out_bias)`，其中 `out_bias=False`）
- Attention和MLP的残差连接是**分开的**：先做Attention残差，再做MLP残差

```python
# ERNIE-Image ErnieImageSharedAdaLNBlock
def forward(self, x, rotary_pos_emb, temb, attention_mask):
    # Attention部分
    attn_out = self.self_attention(x_bsh, ...)
    x = residual + (gate_msa * attn_out)  # Attention残差

    # MLP部分
    residual = x
    x = self.adaLN_mlp_ln(x)
    x = (x * (1 + scale_mlp) + shift_mlp)
    return residual + (gate_mlp * self.mlp(x))  # MLP残差
```

**是否为优点及分析**：
- **更标准的Pre-LN Transformer结构**：ERNIE-Image的"先Attention残差，再MLP残差"的两段式设计是标准的Pre-LN Transformer结构，经过大量研究验证，训练稳定性好
- **FLUX2单流块的并行Attn+MLP设计**：FLUX2的SingleStreamBlock将Attention和MLP的计算**并行化**（通过 `linear1` 同时产出），然后合并投影。这种设计虽然减少了推理延迟（Attn和MLP可以并行计算），但丢失了Attn→MLP的序列依赖关系
- **对生成效果的影响**：ERNIE-Image的串行Attn→MLP设计让MLP可以看到Attention更新后的表示，理论上信息流更完整。这可能对生成质量有微小正面影响

---

### 创新点10：Sequence-first内部表示 vs Batch-first内部表示

**FLUX2 的做法**：
- 全程使用 **Batch-first `[B, S, H]`** 格式

```python
# FLUX2 model.py - forward
img = self.img_in(x)  # [B, L, hidden_size]
txt = self.txt_in(ctx)  # [B, L, hidden_size]
img = torch.cat((txt, img), dim=1)  # dim=1 是序列维度
```

**ERNIE-Image 的做法**：
- 内部主要使用 **Sequence-first `[S, B, H]`** 格式（Megatron风格）
- 在需要做Attention时转为Batch-first，计算后转回

```python
# ERNIE-Image transformer_ernie_image.py
img_sbh = self.x_embedder(hidden_states).transpose(0, 1).contiguous()  # [B,S,H] → [S,B,H]
text_sbh = text_bth.transpose(0, 1).contiguous()
x = torch.cat([img_sbh, text_sbh], dim=0)  # dim=0 是序列维度

# ErnieImageSharedAdaLNBlock中：
x_bsh = x.permute(1, 0, 2)  # [S,B,H] → [B,S,H]，用于Attention
attn_out = attn_out.permute(1, 0, 2)  # [B,S,H] → [S,B,H]，转回
```

**是否为优点及分析**：
- **并行训练优化**：Sequence-first格式是Megatron-LM等大规模分布式训练框架的标准格式，有利于序列并行（Sequence Parallelism）和张量并行（Tensor Parallelism）的实现。对于8B参数的大模型训练，这种格式选择对**训练效率和扩展性**有正面影响
- **这不是生成效果的优化，而是训练和分布式计算效率的优化**
- **推理时的额外开销**：每层都需要做permute操作，增加了少量推理开销

---

### 创新点11：不使用Guidance Embedding，采用标准CFG

**FLUX2 的做法**：
- 有独立的 **`guidance_in`（MLPEmbedder）**，将guidance_scale值编码为嵌入向量加到时间步嵌入上
- 蒸馏版本通过此嵌入实现"隐式CFG"，无需双倍推理

```python
# FLUX2 model.py
self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)
# forward中：
if self.use_guidance_embed:
    guidance_emb = timestep_embedding(guidance, 256)
    vec = vec + self.guidance_in(guidance_emb)
```

**ERNIE-Image 的做法**：
- **没有guidance embedding模块**
- 使用**标准的Classifier-Free Guidance**：正样本和负样本分别做完整前向推理，然后加权合并

```python
# ERNIE-Image pipeline_ernie_image.py
if self.do_classifier_free_guidance:
    latent_model_input = torch.cat([latents, latents], dim=0)  # 复制输入
    # ... 完整前向推理 ...
    pred_uncond, pred_cond = pred.chunk(2, dim=0)
    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)  # 标准CFG
```

**是否为优点及分析**：
- **ERNIE-Image-Turbo的CFG=1.0设计**：虽然标准CFG需要双倍推理开销，但ERNIE-Image的Turbo版本通过DMD+RL蒸馏实现了CFG=1.0（即不需要CFG），此时推理开销与FLUX2的guidance embedding方案相当
- **标准CFG的优点在于更可控**：标准CFG允许在推理时灵活调节guidance_scale的值，而guidance embedding需要模型在训练时见过不同的guidance值
- **这不是生成效果的优化**，而是**推理灵活性和训练简洁性的取舍**。标准CFG更简单直接，但推理时计算量是2倍；guidance embedding更高效但需要额外的训练处理

---

### 创新点12：Final Norm实现差异——无SiLU激活

**FLUX2 的做法**：
- `LastLayer` 使用 `nn.LayerNorm(elementwise_affine=False)` + `SiLU + Linear` 产生shift/scale
- output Linear **不带bias**

```python
# FLUX2 model.py
class LastLayer(nn.Module):
    self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.linear = nn.Linear(hidden_size, out_channels, bias=False)
    self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = (1 + scale) * self.norm_final(x) + shift
        x = self.linear(x)
        return x
```

**ERNIE-Image 的做法**：
- `ErnieImageAdaLNContinuous` 使用 `nn.LayerNorm(elementwise_affine=False)` + **直接Linear**（没有SiLU激活）产生shift/scale
- conditioning向量通过Linear投影后直接chunk成shift/scale，**不经过SiLU激活**
- output Linear **带bias**

```python
# ERNIE-Image transformer_ernie_image.py
class ErnieImageAdaLNContinuous(nn.Module):
    self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)
    self.linear = nn.Linear(hidden_size, hidden_size * 2)  # 注意：带bias

    def forward(self, x, conditioning):
        scale, shift = self.linear(conditioning).chunk(2, dim=-1)  # 无SiLU
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
        return x

# 然后通过独立的 final_linear 输出
self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
nn.init.zeros_(self.final_linear.weight)  # 零初始化
nn.init.zeros_(self.final_linear.bias)    # 零初始化
```

**是否为优点及分析**：
- **final_linear零初始化优点（训练稳定性）**：ERNIE-Image对 `final_linear` 的权重和bias都做了**零初始化**。这意味着训练初期模型输出接近零，去噪预测近似为0，这有助于训练初期的稳定性（避免初期产生过大的预测值导致训练不稳定）
- **无SiLU的设计更简洁**：去掉Final Norm中的SiLU激活减少了非线性变换，使最终的调制过程更"线性"
- **带bias的Linear**：ERNIE-Image的 `self.linear` 和 `self.final_linear` 都带bias，而FLUX2的对应层不带bias。额外的bias参数提供了更多自由度
- **对生成效果的影响**：零初始化策略主要优化训练初期的稳定性，对最终收敛后的生成质量影响有限

---

### 创新点13：时间步嵌入的实现差异

**FLUX2 的做法**：
- 使用自定义的 `timestep_embedding` 函数：`time_factor=1000.0`，先将时间步乘以1000再计算正弦/余弦嵌入
- 嵌入维度为 **256**
- 通过 `MLPEmbedder(256 → hidden_size → hidden_size)` 投影（`Linear → SiLU → Linear`，bias=False）

```python
# FLUX2 model.py
def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    t = time_factor * t  # 乘以1000
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, ...) / half)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding

self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)
```

**ERNIE-Image 的做法**：
- 使用diffusers的 `Timesteps` 模块：`flip_sin_to_cos=False, downscale_freq_shift=0`
- 嵌入维度为 **hidden_size=3072**（而非256！）
- 通过 `TimestepEmbedding(hidden_size → hidden_size → hidden_size)` 投影（`Linear → SiLU → Linear`，带bias）

```python
# ERNIE-Image transformer_ernie_image.py
self.time_proj = Timesteps(hidden_size, flip_sin_to_cos=False, downscale_freq_shift=0)
# Timesteps 使用 num_channels=hidden_size=3072，产生3072维的正弦/余弦嵌入
self.time_embedding = TimestepEmbedding(hidden_size, hidden_size)
# TimestepEmbedding: Linear(3072→3072, bias=True) → SiLU → Linear(3072→3072, bias=True)
```

**是否为优点及分析**：
- **更高维的初始时间嵌入**：ERNIE-Image使用hidden_size（3072）维的正弦/余弦嵌入，远大于FLUX2的256维。更高维的初始嵌入可以为时间步提供更丰富的频率信息
- **带bias的MLP投影**：ERNIE-Image的TimestepEmbedding带bias（`sample_proj_bias=True` 是默认值），而FLUX2的MLPEmbedder不带bias
- **对生成效果的影响**：更高维的时间步嵌入理论上可以让模型更精细地区分不同的噪声水平，可能在**去噪精度**上有微小提升

---

### 创新点14：Attention中AdaLN调制的float32计算精度

**FLUX2 的做法**：
- AdaLN调制直接在当前dtype下计算

```python
# FLUX2 model.py - SingleStreamBlock
x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
```

**ERNIE-Image 的做法**：
- AdaLN调制参数的scale/shift/gate运算都**显式转为float32**计算，然后再转回原dtype

```python
# ERNIE-Image transformer_ernie_image.py - ErnieImageSharedAdaLNBlock
x = (x.float() * (1 + scale_msa.float()) + shift_msa.float()).to(x.dtype)
# ...
x = residual + (gate_msa.float() * attn_out.float()).to(x.dtype)
# ...
x = (x.float() * (1 + scale_mlp.float()) + shift_mlp.float()).to(x.dtype)
return residual + (gate_mlp.float() * self.mlp(x).float()).to(x.dtype)
```

**是否为优点及分析**：
- **数值精度优点（训练稳定性）**：在bf16/fp16混合精度训练中，AdaLN的shift/scale/gate操作涉及乘法和加法，如果在低精度下可能导致数值溢出或精度损失。ERNIE-Image显式使用float32计算这些关键操作，确保数值稳定性
- **这是训练稳定性的优化**，对最终生成效果有间接正面影响（更稳定的训练通常意味着更好的收敛质量）
- **推理开销**：每次float32转换增加了少量计算和内存开销

---

### 创新点15：文本投影的可跳过设计

**FLUX2 的做法**：
- `txt_in` 投影总是执行，无论输入维度是否与隐藏维度匹配

```python
# FLUX2 model.py
self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)
# forward中总是执行：
txt = self.txt_in(ctx)
```

**ERNIE-Image 的做法**：
- 当 `text_in_dim == hidden_size` 时，`text_proj` 设为 **None**，跳过投影

```python
# ERNIE-Image transformer_ernie_image.py
self.text_proj = nn.Linear(text_in_dim, hidden_size, bias=False) if text_in_dim != hidden_size else None

# forward中：
if self.text_proj is not None and text_bth.numel() > 0:
    text_bth = self.text_proj(text_bth)
```

**是否为优点及分析**：
- **灵活性优点**：当文本编码器的输出维度恰好等于模型隐藏维度时，可以省去一个Linear投影层，减少参数量和计算量
- **对生成效果无直接影响**，是一个工程灵活性优化

---

### 创新点16：Attention Mask的显式构建

**FLUX2 的做法**：
- 不使用显式的attention mask（文本token长度固定为512，不需要mask padding）
- 因果注意力通过 `causal_attn_fn` 函数手动分割Q/K/V实现，而非通过mask矩阵

```python
# FLUX2 model.py - causal_attn_fn
# 手动分割Q/K/V，分别计算不同区域的attention
q_txt_img = torch.cat([q_txt, q_img], dim=2)
k_all = torch.cat([k_txt, k_ref, k_img], dim=2)
attn_txt_img = F.scaled_dot_product_attention(q_txt_img, k_all, v_all, is_causal=False)
attn_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=False)  # ref只自注意
```

**ERNIE-Image 的做法**：
- 显式构建 **attention mask**，标记有效token（True）和padding token（False）
- 文本序列长度可变，通过 `text_lens` 参数动态构建mask

```python
# ERNIE-Image transformer_ernie_image.py
valid_text = torch.arange(Tmax, device=device).view(1, Tmax) < text_lens.view(B, 1)
attention_mask = torch.cat([
    torch.ones((B, N_img), device=device, dtype=torch.bool),  # 图像token全部有效
    valid_text  # 文本token根据实际长度mask
], dim=1)[:, None, None, :]
```

**是否为优点及分析**：
- **变长文本支持优点**：ERNIE-Image支持batch中不同样本有不同长度的文本token，通过attention mask正确处理padding。而FLUX2将文本固定为512 token
- **内存效率优点**：变长文本可以避免不必要的padding计算
- **对生成效果的潜在影响**：更精确的mask可以避免模型attend到无意义的padding token，可能对文本-图像对齐有微小正面影响

---

### 创新点17：图像位置ID中嵌入文本长度信息

**FLUX2 的做法**：
- 图像token的位置ID使用固定的4D坐标 `(t=0, h, w, l=0)`，t维度用于区分生成图和参考图
- 文本token的位置ID使用 `(t=0, h=0, w=0, l=序列位置)`

```python
# FLUX2 sampling.py - prc_img
x_coords = {"t": torch.arange(1), "h": torch.arange(h), "w": torch.arange(w), "l": torch.arange(1)}
x_ids = torch.cartesian_prod(x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"])
```

**ERNIE-Image 的做法**：
- 图像token的位置ID格式为 `(text_len, h, w)` —— **第1维使用当前batch样本的实际文本长度**而非时间坐标
- 文本token的位置ID格式为 `(序列位置, 0, 0)`

```python
# ERNIE-Image transformer_ernie_image.py
# 图像token的第1维 = text_lens（文本长度）
image_ids = torch.cat([
    text_lens.float().view(B, 1, 1).expand(-1, N_img, -1),  # 文本长度作为第1维
    grid_yx.view(1, N_img, 2).expand(B, -1, -1)
], dim=-1)

# 文本token的第1维 = 序列位置（0, 1, 2, ...）
text_ids = torch.cat([
    torch.arange(Tmax).view(1, Tmax, 1).expand(B, -1, -1),  # 序列位置
    torch.zeros((B, Tmax, 2))  # h=0, w=0
], dim=-1)
```

**是否为优点及分析**：
- **位置编码中编码了文本长度信息**：这是一个独特的设计——图像token在位置编码的第1维中嵌入了当前batch样本的文本token数量。这意味着同一个图像token在不同长度的文本prompt下会获得不同的位置编码
- **潜在的图文对齐优点**：通过将文本长度信息编码到图像的位置编码中，模型可以隐式感知文本的"信息量"，可能有助于根据文本复杂度调整生成策略
- **在batch维度上区分不同样本的位置编码**：不同batch样本的图像token可以有不同的第1维值（因为text_lens不同），这在FLUX2中是不存在的设计

---

## 三、总结表

| # | 变化创新点 | FLUX2的做法 | ERNIE-Image的做法 | 优化目标 |
|---|-----------|-----------|-----------------|---------|
| 1 | **纯单流架构** | 双流×8 + 单流×48 混合 | 纯单流×24 | 架构简洁性、参数效率、图文深度交互 |
| 2 | **单一全局AdaLN调制模块** | 3个独立Modulation模块 | 1个共享adaLN_modulation + 零初始化 | 参数效率、训练稳定性 |
| 3 | **更紧凑的层数和维度** | 56层, hidden=6144, heads=48 | 24层, hidden=3072, heads=24 | 推理效率、模型规模优化 |
| 4 | **GeGLU FFN（分离gate/up）** | SwiGLU（融合Linear + chunk） | GeGLU（独立gate_proj + up_proj） | FFN表达灵活性 |
| 5 | **3D RoPE + 更多空间维度** | 4D RoPE [32,32,32,32], θ=2000 | 3D RoPE [32,48,48], θ=256 | 空间感知精度（生成效果） |
| 6 | **Conv2d Patch嵌入** | Linear嵌入（外部patchify） | Conv2d嵌入（内部patchify） | 接口封装、工程设计 |
| 7 | **RMSNorm替代LayerNorm** | LayerNorm(affine=False) | RMSNorm(affine=True) | 计算效率、表达能力 |
| 8 | **QKV分离投影** | 融合QKV Linear | 独立to_q/to_k/to_v | 工程灵活性、微调兼容性 |
| 9 | **串行Attn+MLP残差** | 单流块并行Attn+MLP | 标准串行Pre-LN结构 | 信息流完整性 |
| 10 | **Sequence-first格式** | Batch-first [B,S,H] | Sequence-first [S,B,H] | 分布式训练效率 |
| 11 | **标准CFG（无guidance embed）** | Guidance Embedding | 标准CFG双倍推理 | 推理灵活性、训练简洁性 |
| 12 | **Final Norm无SiLU + 零初始化** | SiLU + Linear, no bias | Linear(有bias) + 零初始化 | 训练稳定性 |
| 13 | **高维时间步嵌入** | 256维初始嵌入 | hidden_size(3072)维初始嵌入 | 时间步区分精度（生成效果） |
| 14 | **AdaLN float32精度计算** | 默认dtype计算 | 显式float32计算 | 训练数值稳定性 |
| 15 | **可跳过的文本投影** | 总是投影 | text_in_dim==hidden_size时跳过 | 工程灵活性 |
| 16 | **显式Attention Mask** | 无mask（定长文本） | 动态构建mask（变长文本） | 变长文本支持、内存效率 |
| 17 | **位置编码中嵌入文本长度** | 固定t坐标 | 图像第1维=text_lens | 图文对齐感知（生成效果） |

---

## 四、总体评价

ERNIE-Image的去噪模型相比FLUX2，在网络结构上进行了全面的简化和优化：

1. **架构层面**：以纯单流架构替代双流+单流混合架构，大幅简化了模型结构，减少了工程复杂度，同时让图文交互从第一层开始深度融合
2. **效率层面**：通过RMSNorm、更少的层数、单一调制模块、Sequence-first格式等设计，在推理效率和训练效率上都有优化
3. **训练稳定性层面**：通过AdaLN零初始化、final_linear零初始化、float32精度调制计算等细节设计，增强了训练过程的数值稳定性
4. **位置编码层面**：3D RoPE配合非均匀维度分配和更小的theta，以及位置编码中嵌入文本长度信息，体现了对图像生成任务特点的针对性设计

这些创新点中，**空间维度分配更多的RoPE**、**位置编码中嵌入文本长度信息**、**高维时间步嵌入**等设计直接针对生成效果优化；**纯单流架构**、**紧凑的模型规模**、**RMSNorm**等设计在保持生成质量的同时优化了效率；**零初始化**、**float32精度计算**等设计优化了训练稳定性。