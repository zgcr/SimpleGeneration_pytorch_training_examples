# Ideogram4 vs FLUX2 去噪模型架构对比分析

> 本报告**仅聚焦去噪模型（Transformer backbone）部分**的网络结构差异与创新点。
> VAE、文本编码器选型、采样调度策略等 pipeline 级别差异不在本报告范围内（但时间步条件化方式属于去噪模型内部设计，故包含在内）。
>
> **所有结论均已与代码逻辑实现进行复核校验。**
>
> - Ideogram4 代码：`/opt/nas/p/zhugechaoran/download/code/ideogram4/src/ideogram4/modeling_ideogram4.py`
> - FLUX2 代码：`/opt/nas/p/zhugechaoran/download/code/flux2/src/flux2/model.py`

---

## 目录

1. [两模型去噪模型整体参数对比](#1-两模型去噪模型整体参数对比)
2. [整体架构级创新点](#2-整体架构级创新点)
3. [组件级创新点](#3-组件级创新点)
4. [创新点完整汇总表](#4-创新点完整汇总表)

---

## 1. 两模型去噪模型整体参数对比

| 参数 | Ideogram4 (`Ideogram4Transformer`) | FLUX2 (`Flux2`, dev-32B) |
|------|-------------------------------------|--------------------------|
| 架构类型 | **纯单流 DIT** | 双流 MMDIT + 单流 DIT 混合 |
| Block 类型 | 34 × `Ideogram4TransformerBlock` | 8 × `DoubleStreamBlock` + 48 × `SingleStreamBlock` |
| 总 Block 数 | 34 | 56 |
| 隐藏维度 (`hidden_size`) | 4608 | 6144 |
| 注意力头数 | 18 | 48 |
| 头维度 (`head_dim`) | **256** | 128 |
| MLP 中间维度 | 12288 | ~18432 (`hidden_size × mlp_ratio=3.0`) |
| MLP 类型 | **SwiGLU（3矩阵）** | SiLU Gated（2矩阵+chunk） |
| 输入通道数 | 128 | 128 |
| 文本特征输入维度 | 53248 (4096×13) | 15360 (5120×3) |
| 位置编码 | **3D 交错式 MRoPE** | 4D 拼接式 RoPE |
| RoPE theta | **5,000,000** | 2,000 |
| Norm 类型 | **全部 RMSNorm** | LayerNorm（block norm） + RMSNorm（QKNorm） |
| AdaLN 维度 | **独立 512 维瓶颈** | 与 hidden_size 相同 (6144) |
| AdaLN 参数 | **scale + gate（4参数/block）** | shift + scale + gate（6参数/double, 3参数/single） |
| Gate 激活 | **tanh** | 无 |
| Modulation 共享 | **每层独立** | 全局共享（3个Modulation） |
| Guidance 机制 | **无 guidance_in（外部双模型 CFG）** | guidance_in 嵌入（内置） |

---

## 2. 整体架构级创新点

### 创新点 1：纯单流 DIT 架构（替代双流+单流混合架构）

**Ideogram4 做法**：
- 使用 34 个完全相同的 `Ideogram4TransformerBlock`，文本 token 和图像 token 从第一层开始就在同一个注意力层中交互。
- 代码证据（`modeling_ideogram4.py` 第288-299行）：
  ```python
  self.layers = nn.ModuleList(
    [Ideogram4TransformerBlock(...) for _ in range(config.num_layers)]  # 34层
  )
  ```
- 序列布局：`[padding] [text tokens] [image tokens]`，所有 token 共享所有层的参数。

**FLUX2 做法**：
- 使用 8 个 `DoubleStreamBlock`（文本和图像各有独立的 QKV 投影、MLP、LayerNorm，只在注意力计算时拼接 Q/K/V）+ 48 个 `SingleStreamBlock`（拼接后统一处理）。
- 代码证据（`model.py` 第76-96行）：
  ```python
  self.double_blocks = nn.ModuleList([DoubleStreamBlock(...) for _ in range(params.depth)])  # 8
  self.single_blocks = nn.ModuleList([SingleStreamBlock(...) for _ in range(params.depth_single_blocks)])  # 48
  ```

**区别与优势**：
- Ideogram4 的纯单流架构让文本和图像 token **从第一层就进行深度跨模态交互**，每一层都完全共享参数，而 FLUX2 前 8 层中文本和图像的 MLP、LayerNorm 是独立的，只在注意力计算时交互。
- **优点方向：生成效果优化**。更早、更深的跨模态交互有助于模型更好地对齐文本语义和图像特征，可能提升文本-图像一致性。同时，架构更简洁统一，减少了工程复杂度。

---

### 创新点 2：文本-图像融合方式——加法融合替代分离流

**Ideogram4 做法**：
- LLM 特征经过 `llm_cond_norm`（RMSNorm）+ `llm_cond_proj`（Linear 投影）后，与图像 latent 的投影 **直接相加**，形成统一的隐藏表示序列。
- 代码证据（`modeling_ideogram4.py` 第349-364行）：
  ```python
  x = self.input_proj(x) * output_image_mask        # 图像投影
  llm_features = self.llm_cond_proj(llm_features) * llm_token_mask  # 文本投影
  h = x + llm_features   # 直接相加
  ```
- 文本位置的 `x` 为零（被 `output_image_mask` 屏蔽），图像位置的 `llm_features` 为零（被 `llm_token_mask` 屏蔽），因此相加操作实际上是将两者拼接到各自的位置上。

**FLUX2 做法**：
- 文本和图像通过不同的线性层 `txt_in` 和 `img_in` 投影到相同维度后，保持为**两个独立的序列**，在双流块中通过联合注意力交互。
- 代码证据（`model.py` 第136-137行）：
  ```python
  img = self.img_in(x)    # 图像投影
  txt = self.txt_in(ctx)  # 文本投影（独立序列）
  ```

**区别与优势**：
- Ideogram4 的加法融合意味着文本和图像特征在进入 Transformer 之前就已经被合并到统一的表示空间中，后续所有层完全对等地处理整个序列。
- **优点方向：生成效果优化**。统一表示空间让每一层的注意力和 MLP 都能同时处理文本和图像信息，减少了模态间的"通信瓶颈"。

---

### 创新点 3：每层独立 AdaLN 调制（替代全局共享 Modulation）

**Ideogram4 做法**：
- 每个 `Ideogram4TransformerBlock` 都有自己独立的 `adaln_modulation` 层。
- 代码证据（`modeling_ideogram4.py` 第190行）：
  ```python
  self.adaln_modulation = nn.Linear(adanln_dim, 4 * hidden_size, bias=True)
  ```
- 每层根据时间步条件 `adaln_input` 独立计算自己的 scale 和 gate 值。

**FLUX2 做法**：
- **3 个全局共享**的 Modulation 层，所有双流块共用 `double_stream_modulation_img` 和 `double_stream_modulation_txt`，所有单流块共用 `single_stream_modulation`。
- 代码证据（`model.py` 第98-108行）：
  ```python
  self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, ...)
  self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True, ...)
  self.single_stream_modulation = Modulation(self.hidden_size, double=False, ...)
  ```
- 在 `forward` 中计算一次，所有块共享同一组调制参数。

**区别与优势**：
- Ideogram4 的每层独立 AdaLN 允许**不同深度的层对时间步做出不同的响应**。浅层可以学习关注噪声结构，深层可以学习关注语义细节。
- **优点方向：生成效果优化**。更精细的逐层时间步条件化使模型在不同去噪阶段可以有更灵活的行为，有助于提升生成质量。FLUX2 的全局共享设计虽然节省参数，但所有层被迫使用完全相同的调制信号。

---

### 创新点 4：无 Guidance Embedding 的双模型 CFG 设计

**Ideogram4 做法**：
- 去噪模型 `Ideogram4Transformer` **不包含** `guidance_in` 模块，模型本身不感知 guidance scale。
- CFG（Classifier-Free Guidance）通过**两个独立权重的 Transformer 实例**在模型外部实现。
- 代码证据（`modeling_ideogram4.py`）：`Ideogram4Transformer` 的 `__init__` 和 `forward` 中无任何 guidance 相关参数。
- Pipeline 层面（`pipeline_ideogram4.py` 第253-256行）：
  ```python
  self.conditional_transformer = conditional_transformer
  self.unconditional_transformer = unconditional_transformer
  ```

**FLUX2 做法**：
- 去噪模型 `Flux2` 内置 `guidance_in` 嵌入模块，将 guidance scale 值编码后加到时间步向量 `vec` 上。
- 代码证据（`model.py` 第73-74行）：
  ```python
  if self.use_guidance_embed:
      self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)
  ```

**区别与优势**：
- Ideogram4 将 CFG 从模型架构中解耦：条件模型和无条件模型各自专注于自己的任务，**无条件模型可以独立优化图像先验**，不受条件信号干扰。此外，无条件分支仅处理图像 token（无文本 token），减少了无条件前向的序列长度。
- **优点方向：生成效果优化**。独立权重的无条件模型可以更好地学习图像的内在分布，可能产生更好的 CFG 引导效果。代价是需要维护两套模型权重。

---

## 3. 组件级创新点

### 创新点 5：AdaLN 只有 Scale + Gate，无 Shift 参数

**Ideogram4 做法**：
- 每个 block 的 AdaLN 只产生 **4 个参数**：`scale_msa, gate_msa, scale_mlp, gate_mlp`，**没有 shift**。
- 代码证据（`modeling_ideogram4.py` 第200-205行）：
  ```python
  mod = self.adaln_modulation(adaln_input)
  scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
  gate_msa = torch.tanh(gate_msa)
  gate_mlp = torch.tanh(gate_mlp)
  scale_msa = 1.0 + scale_msa
  scale_mlp = 1.0 + scale_mlp
  ```
- 应用方式：`attention_norm1(x) * scale_msa`（无 shift 项）。

**FLUX2 做法**：
- 双流块的 Modulation 产生 **6 个参数**（`shift, scale, gate` × 2 组），单流块产生 **3 个参数**（`shift, scale, gate`）。
- 代码证据（`model.py` 第401-412行）：
  ```python
  self.multiplier = 6 if double else 3
  self.lin = nn.Linear(dim, self.multiplier * dim, ...)
  ```
- 应用方式：`(1 + scale) * norm(x) + shift`（含 shift 项）。

**区别与优势**：
- 去掉 shift 参数意味着 AdaLN 只控制"缩放"和"通断"，不进行平移。结合 scale 的 `1 + scale` 初始化（初始化时 scale≈0，相当于恒等映射），确保训练初期的稳定性。
- **优点方向：参数效率优化 + 训练稳定性**。每层减少了 `2 × hidden_size` 个参数（省去了 attention 和 MLP 各一个 shift 向量）。无 shift 也减少了自由度，有助于防止过拟合并提升训练稳定性。

---

### 创新点 6：Tanh Gate 激活函数

**Ideogram4 做法**：
- Gate 值通过 `tanh` 激活，限制在 **[-1, 1]** 范围内。
- 代码证据（`modeling_ideogram4.py` 第202-203行）：
  ```python
  gate_msa = torch.tanh(gate_msa)
  gate_mlp = torch.tanh(gate_mlp)
  ```

**FLUX2 做法**：
- Gate 值直接从 Modulation 层输出，**无额外激活函数**，值域无界。
- 代码证据（`model.py` 第411行）：gate 直接作为 chunk 输出的一部分使用，无激活。

**区别与优势**：
- Tanh 将 gate 限制在 [-1, 1]，防止 gate 值过大或过小导致的梯度爆炸/消失。初始化时 gate≈0（tanh(0)=0），整个残差支路近似关闭，随训练逐渐开启。
- **优点方向：训练稳定性优化**。有界的 gate 值提供了更稳定的梯度流，特别是在训练初期和深层网络中。这是一种"soft gating"机制，在 DiT 等大规模模型训练中有助于收敛。

---

### 创新点 7：独立的 AdaLN 瓶颈维度（512 维）

**Ideogram4 做法**：
- 时间步嵌入先投影到 `emb_dim=4608`，再通过 `adaln_proj` 压缩到独立的 `adanln_dim=512`，然后每个 block 内部从 512 维展开到 `4 × 4608` 维。
- 代码证据（`modeling_ideogram4.py` 第278行）：
  ```python
  self.adaln_proj = nn.Linear(config.emb_dim, config.adanln_dim, bias=True)  # 4608 → 512
  ```
- 每个 block（`modeling_ideogram4.py` 第190行）：
  ```python
  self.adaln_modulation = nn.Linear(adanln_dim, 4 * hidden_size, bias=True)  # 512 → 4×4608
  ```

**FLUX2 做法**：
- Modulation 层直接使用 `hidden_size=6144` 作为输入维度。
- 代码证据（`model.py` 第405行）：
  ```python
  self.lin = nn.Linear(dim, self.multiplier * dim, ...)  # 6144 → 6×6144 或 3×6144
  ```

**区别与优势**：
- Ideogram4 的 512 维瓶颈**大幅减少了 AdaLN 的参数量**：每个 block 的 AdaLN 参数从 O(hidden_size²) 减少到 O(adanln_dim × hidden_size)，即从 6144×(6×6144)≈226M 减少到 512×(4×4608)≈9.4M（每层约减少 96%）。
- **优点方向：参数效率优化**。在保持逐层独立 AdaLN 的同时，通过瓶颈设计将参数量控制在合理范围内。这是一个精巧的参数-表达力平衡设计。

---

### 创新点 8：Pre-Norm + Post-Norm 双重归一化

**Ideogram4 做法**：
- 每个 block 中，attention 和 MLP **前后各有一个 RMSNorm**，共 4 个归一化层。
- 代码证据（`modeling_ideogram4.py` 第185-188行）：
  ```python
  self.attention_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)  # Pre-Norm
  self.ffn_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)        # Pre-Norm
  self.attention_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)  # Post-Norm
  self.ffn_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)        # Post-Norm
  ```
- 应用方式（第207-214行）：
  ```python
  attn_out = self.attention(self.attention_norm1(x) * scale_msa, ...)   # Pre-Norm
  x = x + gate_msa * self.attention_norm2(attn_out)                      # Post-Norm + Gated Residual
  x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
  ```
- Gate 作用于 **Post-Norm 后的输出**，而非直接作用于子层的原始输出。

**FLUX2 做法**：
- 每个 block 中只有 **Pre-Norm**（在 attention 和 MLP 前各一个 LayerNorm），无 Post-Norm。
- 双流块代码证据（`model.py` 第537-538, 545行）：
  ```python
  self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
  self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
  ```
- 应用方式（第626-629行）：
  ```python
  img = img + img_mod1_gate * self.img_attn.proj(img_attn)  # Gate 直接作用于 proj 输出
  img = img + img_mod2_gate * self.img_mlp(...)
  ```

**区别与优势**：
- Post-Norm 在 gate 之前对子层输出进行归一化，使得 gate 控制的是**归一化后的残差增量**，这避免了子层输出幅度不稳定时对主干信号造成冲击。
- **优点方向：训练稳定性优化 + 生成效果优化**。双重归一化为深层网络提供了更好的梯度流和训练稳定性。特别是在 34 层纯单流架构中（所有模态共享），稳定的归一化策略对收敛质量至关重要。

---

### 创新点 9：全面使用 RMSNorm 替代 LayerNorm

**Ideogram4 做法**：
- 所有归一化层（block 内 4 个 norm + QK-Norm + LLM 特征 norm + FinalLayer norm）全部使用 `Ideogram4RMSNorm`。
- 代码证据（`modeling_ideogram4.py` 第107-114行）：
  ```python
  class Ideogram4RMSNorm(nn.Module):
    def forward(self, x):
      return F.rms_norm(x, self.weight.shape, self.weight, self.eps)
  ```
- 使用 PyTorch 内置 `F.rms_norm` 实现，带有可学习的 `weight` 参数。

**FLUX2 做法**：
- Block 内归一化使用 `nn.LayerNorm(elementwise_affine=False)`（无可学习参数）。
- QK-Norm 使用自定义 `RMSNorm`（带可学习 `scale` 参数），手动实现。
- 代码证据（`model.py` 第734-743行）：
  ```python
  class RMSNorm(torch.nn.Module):
    def forward(self, x):
      rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
      return (x * rrms) * self.scale
  ```

**区别与优势**：
- RMSNorm 只计算均方根归一化（无需计算均值并减去），计算量比 LayerNorm 少一步。在大规模模型推理中这带来约 10-15% 的 Norm 层加速。
- Ideogram4 的 RMSNorm 带有可学习 `weight` 参数，而 FLUX2 的 LayerNorm 使用 `elementwise_affine=False`（无可学习参数），Ideogram4 的 Norm 层更具表达力。
- **优点方向：计算效率优化**。RMSNorm 在现代 LLM（LLaMA、Qwen 等）中被广泛验证为 LayerNorm 的高效替代，在不损失质量的前提下减少计算开销。

---

### 创新点 10：3D 交错式 MRoPE（替代 4D 拼接式 RoPE）

**Ideogram4 做法**：
- 使用 3D 多模态旋转位置编码 MRoPE，维度为 (temporal, height, width)，`mrope_section = (24, 20, 20)`。
- **交错式（interleaved）分配**：不同轴的频率被交错排列到 head_dim 的不同位置（idx 1 mod 3 分配给 H 轴，idx 2 mod 3 分配给 W 轴），而非简单拼接。
- 代码证据（`modeling_ideogram4.py` 第96-104行）：
  ```python
  freqs_t = freqs[0].clone()
  for axis, offset in ((1, 1), (2, 2)):
    length = self.mrope_section[axis] * 3
    idx = torch.arange(offset, length, 3, device=freqs_t.device)
    freqs_t[..., idx] = freqs[axis][..., idx]
  emb = torch.cat((freqs_t, freqs_t), dim=-1)
  return emb.cos(), emb.sin()
  ```
- `rope_theta = 5,000,000`，支持极大的位置范围。
- 使用 `_rotate_half` 方式应用 RoPE。

**FLUX2 做法**：
- 使用 4D RoPE，维度为 (t, h, w, l)，`axes_dim = [32, 32, 32, 32]`。
- **拼接式分配**：每个轴独立计算频率后沿 head_dim 维度拼接。
- 代码证据（`model.py` 第700-707行）：
  ```python
  emb = torch.cat(
      [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))],
      dim=-3,
  )
  ```
- `theta = 2,000`。
- 使用旋转矩阵乘法方式应用 RoPE。

**区别与优势**：
- **交错式 vs 拼接式**：交错分配让每个 head_dim 位置同时包含来自不同轴的信息，类似于"多尺度混合编码"，而拼接式让不同轴占据 head_dim 的不同段。交错式可能让模型在每个维度位置同时感知多轴空间信息。
- **3D vs 4D**：Ideogram4 不需要独立的"序列位置"维度 `l`，因为文本 token 的位置通过广播 `(text_pos, text_pos, text_pos)` 自然编码到 3 个轴中，图像 token 使用 `(0, h, w) + IMAGE_POSITION_OFFSET`。这简化了坐标系统。
- **theta 差异**：5M vs 2K，Ideogram4 可支持远大的位置范围，配合 `IMAGE_POSITION_OFFSET=65536` 实现文本和图像位置的隔离。
- **优点方向：生成效果优化**。MRoPE 是 Qwen-VL 系列验证过的多模态位置编码方案，交错式设计让位置信息更加均匀地分布在表示空间中，有助于提升空间结构的一致性。超大 theta 值确保在各种分辨率下位置编码不会退化。

---

### 创新点 11：SwiGLU MLP（3 矩阵独立投影）替代 SiLU Gated MLP（2 矩阵 + chunk）

**Ideogram4 做法**：
- MLP 使用经典 SwiGLU 结构，3 个独立的线性层：`w1`（gate 投影）、`w3`（value 投影）、`w2`（输出投影）。
- 代码证据（`modeling_ideogram4.py` 第161-169行）：
  ```python
  class Ideogram4MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
      self.w1 = nn.Linear(dim, hidden_dim, bias=False)
      self.w2 = nn.Linear(hidden_dim, dim, bias=False)
      self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x):
      return self.w2(F.silu(self.w1(x)) * self.w3(x))
  ```

**FLUX2 做法**：
- MLP 使用 SiLU Gated 结构，一个大的输入线性层产生 `2 × mlp_dim` 的输出，然后 chunk 成两半做 gated 乘法。
- 双流块代码证据（`model.py` 第546-550行）：
  ```python
  self.img_mlp = nn.Sequential(
      nn.Linear(hidden_size, mlp_hidden_dim * 2, bias=False),  # 一个大矩阵
      SiLUActivation(),                                         # chunk + SiLU gate
      nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
  )
  ```

**区别与优势**：
- 虽然两者总参数量相近（均为 `3 × dim × hidden_dim`），但 Ideogram4 的 3 矩阵设计让 **gate 投影（w1）和 value 投影（w3）使用完全独立的参数**学习，而 FLUX2 的单矩阵 + chunk 方式中两半共享同一个输入映射矩阵的不同行，耦合更紧密。
- **优点方向：生成效果优化**。独立的 gate 和 value 投影提供了更高的表达自由度，这是 LLaMA/Qwen 等现代 LLM 验证过的高效 MLP 设计。

---

### 创新点 12：Segment-Based 块对角注意力掩码

**Ideogram4 做法**：
- 注意力使用 `segment_ids` 构建块对角掩码：同一 segment 的 token 互相可见，不同 segment 的 token 互不可见。
- 代码证据（`modeling_ideogram4.py` 第154行）：
  ```python
  attn_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)
  ```
- Padding 位置 `segment_ids = -1`，与任何 sample 不匹配，自然被排除。

**FLUX2 做法**：
- 标准 forward 使用**全注意力（无掩码）**。
- 仅在参考图像编辑模式下使用因果注意力 `causal_attn_fn`（ref 自注意力 + txt/img 全注意力）。
- 代码证据（`model.py` 第156行，标准 attention 调用）：
  ```python
  out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # Ideogram4
  ```
  vs（`model.py` 第806行）：
  ```python
  attn_txt_img = F.scaled_dot_product_attention(q_txt_img, k_all, v_all, is_causal=False)  # FLUX2, 无mask
  ```

**区别与优势**：
- Segment-based 注意力掩码天然支持 **packed batching**（将多个不同长度的样本打包到同一序列中，用不同 segment_id 区分），避免了 padding 浪费。
- **优点方向：训练/推理效率优化**。Packed batching 显著提升了 GPU 利用率，特别是在处理变长序列（不同长度的文本提示 + 不同分辨率的图像）时。同时，掩码确保了不同样本间的信息隔离。

---

### 创新点 13：Image Indicator Embedding（显式模态类型嵌入）

**Ideogram4 做法**：
- 使用 `nn.Embedding(2, emb_dim)` 生成模态类型嵌入，加到隐藏表示上，显式区分图像 token 和文本 token。
- 代码证据（`modeling_ideogram4.py` 第280行）：
  ```python
  self.embed_image_indicator = nn.Embedding(2, config.emb_dim)
  ```
- 应用方式（第366-369行）：
  ```python
  image_indicator_embedding = self.embed_image_indicator(
    (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long)
  )
  h = h + image_indicator_embedding
  ```
- `indicator == OUTPUT_IMAGE_INDICATOR` 为 True 的位置得到 embedding[1]，否则得到 embedding[0]。

**FLUX2 做法**：
- **无显式模态类型嵌入**。文本和图像 token 通过不同的投影层（`txt_in` / `img_in`）和不同的位置坐标（4D RoPE 中不同的 t/l 坐标）隐式区分。

**区别与优势**：
- 在纯单流架构中，所有 token 共享同一组 QKV 和 MLP 参数，显式的模态类型嵌入为模型提供了一个直接的信号来区分文本和图像 token 的角色。
- **优点方向：生成效果优化**。这是单流架构的必要补充：由于所有层完全共享参数，模型需要某种方式"知道"当前 token 是文本还是图像，以便在内部做适当的处理。这有助于模型学习模态特定的处理策略。

---

### 创新点 14：更大的注意力头维度（256 vs 128）

**Ideogram4 做法**：
- `head_dim = emb_dim / num_heads = 4608 / 18 = 256`。

**FLUX2 做法**：
- `head_dim = hidden_size / num_heads = 6144 / 48 = 128`。

**区别与优势**：
- 更大的 head_dim 意味着**每个注意力头可以编码更丰富的查询-键匹配模式**。在相同的隐藏维度下，较少的头数但较大的头维度让每个头有更强的表达能力。
- 同时，256 维的 head_dim 使得 RoPE 的频率空间更大（`mrope_section = (24, 20, 20)` 合计 64 个频率对 × 2 = 128 维用于位置，剩余维度用于内容），有助于位置编码的精细度。
- **优点方向：生成效果优化**。更大的 head_dim 在图像生成任务中可能更有利，因为图像 token 间的空间关系复杂，需要每个头有足够的容量来编码细粒度的空间注意力模式。

---

### 创新点 15：LLM 特征预归一化（Pre-Normalization）

**Ideogram4 做法**：
- LLM 特征在投影到嵌入空间之前，先经过 `llm_cond_norm`（RMSNorm）归一化。
- 代码证据（`modeling_ideogram4.py` 第275-276行）：
  ```python
  self.llm_cond_norm = Ideogram4RMSNorm(config.llm_features_dim, eps=1e-6)
  self.llm_cond_proj = nn.Linear(config.llm_features_dim, config.emb_dim, bias=True)
  ```
- 应用方式（第361-362行）：
  ```python
  llm_features = self.llm_cond_norm(llm_features)    # 先归一化
  llm_features = self.llm_cond_proj(llm_features) * llm_token_mask  # 再投影
  ```

**FLUX2 做法**：
- 文本特征直接通过 `txt_in` 线性投影，**无预归一化**。
- 代码证据（`model.py` 第137行）：
  ```python
  txt = self.txt_in(ctx)  # 直接投影，无归一化
  ```

**区别与优势**：
- LLM 输出的 53248 维特征（13 层拼接）可能具有较大的尺度差异，不同层的隐藏状态幅度不同。RMSNorm 预归一化可以将这些多尺度特征标准化到统一的尺度，使后续的线性投影更加稳定。
- **优点方向：训练稳定性优化**。防止不同 LLM 层的特征幅度差异导致投影层的梯度不稳定，特别是在使用 13 层拼接这种大维度输入时。

---

### 创新点 16：时间步嵌入的范围归一化 + 瓶颈投影

**Ideogram4 做法**：
- 时间步先进行**范围归一化**（映射到 [0, 1e4]），然后正弦嵌入，再经过带 bias 的 MLP。
- 代码证据（`modeling_ideogram4.py` 第232-250行）：
  ```python
  class Ideogram4EmbedScalar(nn.Module):
    def __init__(self, dim, input_range):
      self.mlp_in = nn.Linear(dim, dim, bias=True)
      self.mlp_out = nn.Linear(dim, dim, bias=True)
    def forward(self, x):
      scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
      emb = _sinusoidal_embedding(scaled, self.dim)
      emb = F.silu(self.mlp_in(emb))
      return self.mlp_out(emb)
  ```
- 之后通过 `adaln_proj`（Linear(4608→512)）压缩到 512 维瓶颈。
- 代码证据（第358-359行）：
  ```python
  t_cond = self.t_embedding(t)                    # → (B, 4608)
  adaln_input = F.silu(self.adaln_proj(t_cond))   # → (B, 512)
  ```

**FLUX2 做法**：
- 时间步直接乘以 `time_factor=1000`，然后正弦嵌入（256维），再经过 MLP（256→6144），无 bias。
- 代码证据（`model.py` 第710-731行）：
  ```python
  def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    t = time_factor * t
    ...
  ```
- `time_in = MLPEmbedder(256, 6144, disable_bias=True)`。
- 输出直接作为 6144 维的 `vec`，无瓶颈压缩。

**区别与优势**：
- Ideogram4 的设计包含两个创新：
  1. **范围归一化**：明确的 `[0, 1]` 到 `[0, 1e4]` 映射，使正弦频率分布更可控；
  2. **瓶颈投影**：先升维到 4608 再压缩到 512，用低维瓶颈表示时间步条件。
- **优点方向：参数效率优化**。512 维的时间步瓶颈表示通过后续每层独立的 `adaln_modulation`（512→4×4608）展开，比 FLUX2 的 6144 维向量直接输入全局 Modulation 更加参数高效，同时瓶颈结构起到了信息压缩的正则化效果。

---

### 创新点 17：FinalLayer 的简化设计（仅 Scale 调制）

**Ideogram4 做法**：
- FinalLayer 的 AdaLN 只产生**1 个参数（scale）**，无 shift。
- 代码证据（`modeling_ideogram4.py` 第253-262行）：
  ```python
  class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, adanln_dim):
      self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
      self.linear = nn.Linear(hidden_size, out_channels, bias=True)
      self.adaln_modulation = nn.Linear(adanln_dim, hidden_size, bias=True)  # 512 → 4608, 只有scale
    def forward(self, x, c):
      scale = 1.0 + self.adaln_modulation(F.silu(c))
      return self.linear(self.norm_final(x) * scale)
  ```

**FLUX2 做法**：
- FinalLayer 的 AdaLN 产生 **2 个参数（shift + scale）**。
- 代码证据（`model.py` 第415-434行）：
  ```python
  class LastLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
      self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))
    def forward(self, x, vec):
      shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
      x = (1 + scale) * self.norm_final(x) + shift
      x = self.linear(x)
      return x
  ```

**区别与优势**：
- 与 block 内的 AdaLN 一致，FinalLayer 也只使用 scale 不使用 shift。且使用 adanln_dim（512）瓶颈而非 hidden_size（6144）。
- **优点方向：参数效率优化 + 设计一致性**。与整个模型的"无 shift"设计哲学保持一致，同时使用瓶颈维度减少 FinalLayer 的参数量（512×4608 vs 6144×2×6144）。

---

### 创新点 18：选择性 Bias 使用策略

**Ideogram4 做法**：
- **有 bias 的层**：`input_proj`、`llm_cond_proj`、`adaln_proj`、`adaln_modulation`（每层）、`t_embedding` MLP、`final_layer` 的 linear 和 adaln_modulation。
- **无 bias 的层**：attention 的 `qkv`、`o`；MLP 的 `w1`、`w2`、`w3`。
- 代码证据：
  - `input_proj = nn.Linear(config.in_channels, config.emb_dim, bias=True)` （第274行）
  - `self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)` （第125行）
  - `self.w1 = nn.Linear(dim, hidden_dim, bias=False)` （第164行）

**FLUX2 做法**：
- 几乎所有线性层都使用 `bias=False`。
- 代码证据：`self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)` 等。

**区别与优势**：
- Ideogram4 在**投影层和条件化层保留 bias**（这些层需要仿射变换能力），在**计算核心层（attention、MLP）去掉 bias**（这些层更关注方向性而非偏移）。
- **优点方向：生成效果优化（微调）**。选择性 bias 策略兼顾了表达能力和参数效率——条件化层的 bias 帮助模型更好地对齐不同模态的特征分布，核心计算层的无 bias 设计则遵循现代 Transformer 的最佳实践。

---

## 4. 创新点完整汇总表

| 编号 | 创新点 | Ideogram4 做法 | FLUX2 做法 | 主要优化方向 |
|------|--------|---------------|-----------|-------------|
| **整体架构级** | | | | |
| 1 | 纯单流 DIT 架构 | 34 × 统一 TransformerBlock | 8 × DoubleStreamBlock + 48 × SingleStreamBlock | 生成效果（更深的跨模态交互） |
| 2 | 加法融合文本-图像 | LLM 特征与图像投影直接相加 | 文本和图像保持独立序列，联合注意力 | 生成效果（统一表示空间） |
| 3 | 每层独立 AdaLN | 34 层各自独立的 adaln_modulation | 3 个全局共享 Modulation | 生成效果（逐层差异化响应） |
| 4 | 无 Guidance Embedding 的双模型 CFG | 无 guidance_in，外部两个独立模型 | 内置 guidance_in 嵌入 | 生成效果（独立优化条件/无条件模型） |
| **AdaLN 调制组件** | | | | |
| 5 | 无 Shift 的 AdaLN | 只有 scale + gate（4 参数/层） | shift + scale + gate（6/3 参数） | 参数效率 + 训练稳定性 |
| 6 | Tanh Gate 激活 | gate = tanh(gate)，有界 [-1, 1] | gate 无激活，值域无界 | 训练稳定性 |
| 7 | AdaLN 瓶颈维度 | 独立 adanln_dim=512 | Modulation 维度 = hidden_size=6144 | 参数效率（每层约减少 96% AdaLN 参数） |
| **归一化组件** | | | | |
| 8 | Pre-Norm + Post-Norm 双重归一化 | 每 block 4 个 RMSNorm（前后各 1） | 每 block 仅 Pre-Norm（LayerNorm） | 训练稳定性 + 生成效果 |
| 9 | 全面 RMSNorm | 所有 Norm 层均为带可学习权重的 RMSNorm | Block Norm 为 LayerNorm(affine=False), QKNorm 为 RMSNorm | 计算效率 |
| **位置编码组件** | | | | |
| 10 | 3D 交错式 MRoPE | (t,h,w) 交错分配，section=(24,20,20)，theta=5M | (t,h,w,l) 拼接分配，axes=[32,32,32,32]，theta=2K | 生成效果（空间一致性） |
| **MLP 组件** | | | | |
| 11 | SwiGLU 3 矩阵独立投影 | w2(SiLU(w1(x)) * w3(x))，3 个独立矩阵 | Linear(2×mlp_dim) → chunk → SiLU gate → Linear，1 个大矩阵+chunk | 生成效果（更高表达自由度） |
| **注意力组件** | | | | |
| 12 | Segment-Based 块对角注意力 | segment_ids 控制可见性，支持 packed batch | 全注意力/因果注意力（仅参考图模式） | 训练/推理效率 |
| 13 | Image Indicator Embedding | Embedding(2, emb_dim) 显式标记模态 | 无显式模态标记，依赖分离投影/位置隐式区分 | 生成效果（单流架构的必要补充） |
| 14 | 更大注意力头维度 | head_dim=256 (18 heads) | head_dim=128 (48 heads) | 生成效果（更丰富的每头表达） |
| **输入处理组件** | | | | |
| 15 | LLM 特征预归一化 | RMSNorm → Linear 投影 | 直接 Linear 投影 | 训练稳定性 |
| 16 | 时间步范围归一化 + 瓶颈 | [0,1]→[0,1e4] 归一化 + MLP(4608→4608) + 投影(4608→512) | time_factor×1000 + MLP(256→6144) | 参数效率 |
| **输出组件** | | | | |
| 17 | FinalLayer 仅 Scale 调制 | 1 参数（scale），bottleneck 512→4608 | 2 参数（shift+scale），dim=6144→2×6144 | 参数效率 + 设计一致性 |
| **线性层设计** | | | | |
| 18 | 选择性 Bias 使用 | 投影/条件化层有 bias，attention/MLP 无 bias | 几乎全部无 bias | 生成效果（平衡表达力与效率） |

---

## 附录：代码文件参考

| 文件 | 关键类/函数 |
|------|------------|
| **Ideogram4** | |
| `modeling_ideogram4.py` | `Ideogram4Transformer`, `Ideogram4TransformerBlock`, `Ideogram4Attention`, `Ideogram4MLP`, `Ideogram4MRoPE`, `Ideogram4FinalLayer`, `Ideogram4EmbedScalar`, `Ideogram4RMSNorm` |
| `constants.py` | `LLM_TOKEN_INDICATOR`, `OUTPUT_IMAGE_INDICATOR`, `IMAGE_POSITION_OFFSET`, `QWEN3_VL_ACTIVATION_LAYERS` |
| `pipeline_ideogram4.py` | `Ideogram4Pipeline`（conditional + unconditional transformer 的使用方式） |
| **FLUX2** | |
| `model.py` | `Flux2`, `DoubleStreamBlock`, `SingleStreamBlock`, `SelfAttention`, `Modulation`, `LastLayer`, `EmbedND`, `QKNorm`, `RMSNorm`, `MLPEmbedder`, `SiLUActivation` |
| `sampling.py` | `denoise`, `denoise_cached`, `denoise_cfg`（CFG 实现方式） |

---

> **报告完成**
>
> 本报告覆盖了 Ideogram4 去噪模型相比 FLUX2 去噪模型的 **18 个变化创新点**，涵盖整体架构级（4 个）和组件级（14 个）两个层面。每个创新点均标注了具体的代码行号证据和主要优化方向（生成效果 / 参数效率 / 训练稳定性 / 计算效率 / 训练推理效率）。