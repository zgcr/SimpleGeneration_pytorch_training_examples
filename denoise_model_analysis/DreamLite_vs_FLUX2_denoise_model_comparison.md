# DreamLite vs FLUX2 去噪模型架构对比分析

> 本文档基于对 `/opt/nas/p/zhugechaoran/download/code/DreamLite/` 和 `/opt/nas/p/zhugechaoran/download/code/flux2/` 两个代码仓库中去噪模型部分源代码的逐行分析完成。每条结论均已与代码实现交叉验证。

---

## 一、整体架构概览

### DreamLite 去噪模型：`DreamLiteUNetModel`
- **文件位置**：`dreamlite/models/unets/unet_2d_condition_mobile.py`
- **架构类型**：经典 UNet（Encoder-Bottleneck-Decoder + Skip Connections）
- **参数量**：0.39B
- **核心组件**：Conv_in → Down Blocks (ResBlock + CrossAttn Transformer) → Mid Block → Up Blocks (ResBlock + CrossAttn Transformer + Skip Connection) → Conv_out

### FLUX2 去噪模型：`Flux2`
- **文件位置**：`src/flux2/model.py`
- **架构类型**：Transformer-only（双流 MMDIT + 单流 DIT）
- **参数量**：~32B（Dev版），~9B/~4B（Klein版）
- **核心组件**：img_in/txt_in → DoubleStreamBlock ×8 → SingleStreamBlock ×48 → final_layer

---

## 二、整体架构层面的变化创新点

### 创新点 1：UNet 架构替代 Transformer-only 架构

**FLUX2 的做法**：
- 使用纯 Transformer 架构，由 `DoubleStreamBlock`（双流 MMDIT）×8 + `SingleStreamBlock`（单流 DIT）×48 组成
- 图像 latent 被 flatten 成 1D token 序列（`[B, H/16*W/16, 128]`），完全丢弃了 2D 空间结构
- 代码证据（`model.py` L68-96）：
  ```python
  self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
  self.double_blocks = nn.ModuleList([DoubleStreamBlock(...) for _ in range(params.depth)])
  self.single_blocks = nn.ModuleList([SingleStreamBlock(...) for _ in range(params.depth_single_blocks)])
  ```

**DreamLite 的做法**：
- 使用经典 UNet 架构，包含 Encoder（下采样路径）、Bottleneck（中间块）、Decoder（上采样路径）和 Skip Connections
- 保留 2D 空间结构，通过卷积操作直接处理 `[B, C, H, W]` 格式的特征图
- 代码证据（`unet_2d_condition_mobile.py` L177-184, L318-319）：
  ```python
  down_block_types = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")
  mid_block_type = "UNetMidBlock2DCrossAttn"
  up_block_types = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")
  self.down_blocks = nn.ModuleList([])
  self.up_blocks = nn.ModuleList([])
  ```

**优点与分析**：
- **参数效率极高**：0.39B vs 32B，缩小约 80 倍，优化维度为**模型大小和推理效率**
- **保留空间归纳偏置**：卷积操作天然保持局部空间相关性和平移不变性，UNet 的 Skip Connections 有助于保持高频细节（如边缘、纹理），在小参数量下可能生成效果更好
- **端侧部署友好**：卷积操作在移动端芯片（如 Apple Neural Engine）上有成熟的硬化加速支持，优化维度为**部署效率**
- **不一定提升大规模生成质量**：在参数量充足的情况下，Transformer 的全局注意力机制理论上能捕获更复杂的长距离依赖关系，但 UNet 在小模型场景下的性价比更高

---

### 创新点 2：In-Context Spatial Concatenation（空间维度拼接）统一生成与编辑

**FLUX2 的做法**：
- 将参考图的 VAE latent tokens 在**序列维度（sequence dimension）**拼接到噪声 latent tokens 前面，形成 `[ref_tokens, img_tokens]` 的序列
- 使用**因果注意力**（`causal_attn_fn`）控制信息流方向：参考图 tokens 只能自注意力，生成图 tokens 可以看到所有 tokens
- 代码证据（`model.py` L192-193, L758-815）：
  ```python
  x = torch.cat([x_seq_concat, x], dim=1)  # 序列维度拼接
  # causal_attn_fn: ref只能自注意力，img可以看到所有
  attn_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)
  attn_txt_img = F.scaled_dot_product_attention(q_txt_img, k_all, v_all)
  ```

**DreamLite 的做法**：
- 将噪声 latent 和条件图像 latent 在**宽度维度（width dimension, dim=3）**拼接，形成 `[B, C, H, W*2]` 的特征图
- 文生图时条件图像 latent 为全零张量，图像编辑时为源图像的 VAE 编码结果
- UNet 预测后裁剪前 W 宽度部分作为最终输出
- 代码证据（`pipeline_dreamlite.py` L399, L405, L419）：
  ```python
  model_input = torch.cat([latents_in, cond_img_in], dim=3)  # 宽度维度拼接
  noise_pred = noise_pred[..., :latents.shape[-1]]  # 裁剪回原始宽度
  ```

**优点与分析**：
- **架构统一性**：一个 UNet、一次 forward pass 同时处理生成和编辑任务，无需像 FLUX2 那样区分有无参考图的不同 forward 路径（`forward` vs `forward_kv_extract` vs `forward_kv_cached`）
- **卷积操作天然适配**：空间维度拼接完美利用了卷积的局部感受野，相邻的源图像像素和目标像素在空间上相邻，卷积核可以直接比较它们，可能提升编辑任务的**空间对齐能力**
- **简单高效**：不需要因果注意力掩码等复杂机制，实现更简洁

---

### 创新点 3：双重 Classifier-Free Guidance（三路 CFG）

**FLUX2 的做法**：
- 使用 **Guidance Embedding** 方式：将 guidance 值通过 `MLPEmbedder` 编码后加到时间步嵌入上
- 蒸馏版模型（Klein 系列）`use_guidance_embed=False`，不使用 guidance
- Base 模型支持标准 CFG（`denoise_cfg` 函数，双路）
- 代码证据（`model.py` L73-74, L128-130）：
  ```python
  self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)
  vec = vec + self.guidance_in(guidance_emb)
  ```

**DreamLite 的做法**：
- 图像编辑时采用**三路 CFG**：同时计算无条件预测、图像条件预测、完整条件预测
- 使用两个独立的引导系数分别控制文本引导和图像保持
- 代码证据（`pipeline_dreamlite.py` L403-406, L424-427）：
  ```python
  latents_in = torch.cat([latents] * 3)
  cond_img_in = torch.cat([uncond_image_latents, image_latents, image_latents])
  # ...
  noise_pred = noise_pred_uncond + \
      guidance_scale * (noise_pred_text - noise_pred_image) + \
      image_guidance_scale * (noise_pred_image - noise_pred_uncond)
  ```

**优点与分析**：
- **精细化控制**：可以独立调节 `guidance_scale`（文本引导强度）和 `image_guidance_scale`（图像保持强度），在编辑任务中可以更精确地平衡"忠实于文本指令"和"保持源图像一致性"两个目标，**可能提升编辑效果**
- **更直接的解耦**：FLUX2 的 guidance embedding 只是一个标量值的隐式注入，无法像 DreamLite 这样显式解耦文本和图像的引导方向
- **计算代价**：编辑模式需要 3 次 forward pass（vs FLUX2 的 1-2 次），是效率的折中

---

## 三、组件层面的变化创新点

### 创新点 4：深度可分离卷积（Depthwise Separable Convolution）

**FLUX2 的做法**：
- 不使用任何卷积操作，全部为线性层（`nn.Linear`）和注意力操作
- 代码证据（`model.py`）：所有层均为 `nn.Linear`

**DreamLite 的做法**：
- 在 `ResnetBlock2D` 中支持通过 `use_sep_conv=True` 参数将标准 3×3 卷积替换为**深度可分离卷积**（Depthwise Conv + Pointwise Conv）
- 深度可分离卷积还支持 `expand_ratio` 参数，在深度卷积和逐点卷积之间进行通道扩展
- 代码证据（`resnet.py` L188-207, L289-293）：
  ```python
  class DepthwiseSeparableConv(nn.Module):
      def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, expand_ratio=1):
          self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=in_channels, bias=bias)
          self.pointwise = nn.Conv2d(in_channels, int(out_channels*expand_ratio), kernel_size=1, bias=bias)
  ```
  ```python
  if use_sep_conv:
      expand_ratio = 2
      self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3,
                                          stride=1, padding=1, expand_ratio=expand_ratio)
  ```

**优点与分析**：
- 标准 3×3 卷积参数量为 `in_channels × out_channels × 9`，深度可分离卷积参数量约为 `in_channels × 9 + in_channels × out_channels`，参数量减少约 `9/(9+out_channels)` 倍
- 优化维度为**模型参数量和计算效率**，对生成质量影响较小

---

### 创新点 5：分组查询注意力（GQA, Grouped Query Attention）

**FLUX2 的做法**：
- 使用标准多头注意力（MHA），Q/K/V 头数相同
- 代码证据（`model.py` L384）：
  ```python
  self.qkv = nn.Linear(dim, dim * 3, bias=False)  # Q, K, V 维度相同
  ```

**DreamLite 的做法**：
- 支持通过 `num_kv_heads` 参数配置 GQA，KV 头数可以少于 Q 头数
- 该参数从 `DreamLiteUNetModel` 逐层传递到 `BasicTransformerBlock` → `Attention`，同时作用于 Self-Attention 和 Cross-Attention
- 代码证据（`attention_processor.py` L110-111, L144, L246-251）：
  ```python
  class Attention(nn.Module):
      def __init__(self, ..., kv_heads: Optional[int] = None, ...):
          self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
          self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
          self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
  ```
  `attention.py` L325, L400-401, L424-434：
  ```python
  class BasicTransformerBlock(nn.Module):
      def __init__(self, ..., num_kv_heads: Optional[int] = None):
          self.attn1 = Attention(..., kv_heads=num_kv_heads)  # Self-Attention
          self.attn2 = Attention(..., kv_heads=num_kv_heads)  # Cross-Attention
  ```

**优点与分析**：
- 当 `kv_heads < heads` 时，KV 投影的参数量和推理时的 KV cache 内存占用线性减少
- 多个 Q 头共享同一组 KV，在大量实践中（如 LLaMA 2/3）证明对模型质量影响极小
- 优化维度为**内存效率和推理速度**

---

### 创新点 6：可配置的 FFN 维度乘子（ff_mult）

**FLUX2 的做法**：
- FFN（MLP）的隐藏层维度通过 `mlp_ratio=3.0` 固定设置
- 使用 SiLU Gated 激活（SwiGLU），实际中间维度为 `hidden_size × mlp_ratio × 2`（因为 gating 需要双倍宽度，但输出通过 gating 降回）
- 代码证据（`model.py` L450-451）：
  ```python
  self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
  self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim * self.mlp_mult_factor, bias=False)
  ```

**DreamLite 的做法**：
- 通过 `ff_mult` 参数（默认值 4）灵活控制 `BasicTransformerBlock` 中 `FeedForward` 的隐藏层维度
- 该参数从 `DreamLiteUNetModel` → `get_down_block/get_mid_block/get_up_block` → `Transformer2DModel` → `BasicTransformerBlock` → `FeedForward` 逐层传递
- 代码证据（`attention.py` L459-467）：
  ```python
  self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout, activation_fn=activation_fn, ...)
  ```
  `unet_2d_condition_mobile.py` L226-228：
  ```python
  ff_mult: int = 4,
  num_kv_heads: int = None,
  num_mid_layers: int = 1
  ```

**优点与分析**：
- 允许在不改变注意力头数或隐藏维度的情况下，单独缩减 FFN 的计算量
- 例如将 `ff_mult` 从 4 降到 2，FFN 参数量和计算量减半
- 优化维度为**灵活的模型容量调控**，在模型压缩和效率优化中有用

---

### 创新点 7：可选移除 Self-Attention（use_self_attention 开关）

**FLUX2 的做法**：
- Self-Attention 是所有 block 的固有组件，无法单独移除
- DoubleStreamBlock 中图像和文本各有独立的 Self-Attention（联合计算）
- SingleStreamBlock 中拼接后的序列整体做 Self-Attention
- 代码证据（`model.py` L524-567, L437-521）：所有 block 必须执行注意力

**DreamLite 的做法**：
- `BasicTransformerBlock` 支持通过 `use_self_attention=False` 关闭自注意力层
- 关闭时直接跳过 Self-Attention，仅保留 Cross-Attention 和 FFN
- 同时提供了专用的 block 类型 `CrossAttnDownRemoveSelfAttnBlock2D` 和 `CrossAttnUpRemoveSelfAttnBlock2D`
- 代码证据（`attention.py` L323, L389-403, L532-549）：
  ```python
  use_self_attention: bool = True,
  # ...
  if self.use_self_attention:
      self.attn1 = Attention(query_dim=dim, ...)
  else:
      self.attn1 = None
  # forward:
  if self.use_self_attention:
      attn_output = self.attn1(norm_hidden_states, ...)
      hidden_states = attn_output + hidden_states
  else:
      hidden_states = norm_hidden_states  # 直接跳过
  ```

**优点与分析**：
- Self-Attention 的计算复杂度为 O(N²)，其中 N 为空间 token 数。在高分辨率下（如 1024×1024，latent 128×128），Self-Attention 是计算瓶颈
- 移除 Self-Attention 后仅保留 Cross-Attention（复杂度取决于文本序列长度，通常远小于图像 token 数），大幅减少计算量
- 在 UNet 中，卷积操作已经提供了局部空间建模能力，Self-Attention 的全局建模可部分被 skip connections 和多层卷积补偿
- 优化维度为**推理速度**，可能略影响需要全局一致性的生成质量

---

### 创新点 8：Token Refiner 机制（文本特征精炼）

**FLUX2 的做法**：
- 文本编码器输出直接通过单个 `nn.Linear` 投影到模型隐藏维度，无额外精炼
- 代码证据（`model.py` L70）：
  ```python
  self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)
  ```

**DreamLite 的做法**：
- 支持多种文本 token 精炼方式，通过 `encoder_hid_dim_type` 参数配置：
  - `"text_proj"`：简单线性投影（与 FLUX2 类似）
  - `"text_proj_rms"`：线性投影 + **RMSNorm 归一化**
  - `"text_token_refiner"`：使用 `HunyuanVideoTokenRefiner`（2 层 Transformer + MLP，注意力头维度 128）
  - `"light_text_token_refiner"`：轻量版 TokenRefiner（1 层，固定 1792 宽度，mlp_ratio=3）
  - `"large_text_token_refiner"`：大型版 TokenRefiner（6 层）
- 代码证据（`unet_2d_condition_mobile.py` L599-632）：
  ```python
  elif encoder_hid_dim_type == "text_proj_rms":
      self.encoder_hid_proj = nn.Sequential(
          nn.Linear(encoder_hid_dim, cross_attention_dim),
          RMSNorm(cross_attention_dim, eps=1e-5, elementwise_affine=True),
      )
  elif encoder_hid_dim_type == "text_token_refiner":
      self.encoder_hid_proj = HunyuanVideoTokenRefiner(
          in_channels=encoder_hid_dim, num_attention_heads=cross_attention_dim // 128,
          attention_head_dim=128, num_layers=2,
      )
  elif encoder_hid_dim_type == "light_text_token_refiner":
      self.encoder_hid_proj = HunyuanVideoTokenRefiner(
          in_channels=encoder_hid_dim, num_attention_heads=1792 // 128,
          attention_head_dim=128, num_layers=1, mlp_ratio=3
      )
  elif encoder_hid_dim_type == "large_text_token_refiner":
      self.encoder_hid_proj = HunyuanVideoTokenRefiner(
          in_channels=encoder_hid_dim, num_attention_heads=cross_attention_dim // 128,
          attention_head_dim=128, num_layers=6,
      )
  ```

**优点与分析**：
- Token Refiner 对文本编码器的输出进行二次处理，可以：
  1. 对齐文本特征与图像特征的分布（弥补文本编码器与去噪模型训练目标不一致的问题）
  2. 在文本 tokens 之间进行自注意力交互，增强上下文建模
  3. 通过多种配置适配不同的模型规模需求
- `text_proj_rms` 中添加 RMSNorm 可以稳定文本特征的数值范围，**可能提升训练稳定性和生成质量**
- 多层 TokenRefiner **可能提升文本理解的细粒度**，尤其对复杂提示词有益

---

### 创新点 9：Cross-Attention 分离式文本条件注入（vs 联合注意力）

**FLUX2 的做法**：
- 在 DoubleStreamBlock 中，图像和文本的 Q/K/V 被**拼接后联合计算注意力**，然后再分开
- 在 SingleStreamBlock 中，图像和文本 tokens **直接拼接为一个序列**，用同一套参数处理
- 代码证据（`model.py` L594-596）：
  ```python
  q = torch.cat((txt_q, img_q), dim=2)
  k = torch.cat((txt_k, img_k), dim=2)
  v = torch.cat((txt_v, img_v), dim=2)
  ```

**DreamLite 的做法**：
- 使用经典的**分离式 Cross-Attention**：`BasicTransformerBlock` 中 Self-Attention 处理图像特征，Cross-Attention 将文本特征作为 K/V、图像特征作为 Q 进行交互
- 图像和文本特征始终通过不同的注意力层处理，各自有独立的归一化和投影
- 代码证据（`attention.py` L559-583）：
  ```python
  # Cross-Attention: Q来自图像，K/V来自文本
  attn_output = self.attn2(
      norm_hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=encoder_attention_mask,
  )
  hidden_states = attn_output + hidden_states
  ```

**优点与分析**：
- Cross-Attention 将图像和文本的角色**显式区分**（图像为 Query，文本为 Key/Value），语义更清晰
- 计算效率更高：Cross-Attention 的复杂度为 O(N_img × N_txt)，而联合注意力的复杂度为 O((N_img + N_txt)²)
- FLUX2 的联合注意力允许文本 tokens 也能 attend 到图像 tokens（双向交互），理论上信息流更丰富，但在小模型场景下 Cross-Attention 的性价比更高
- 优化维度为**计算效率**，且在 UNet 这种卷积为主的架构中是自然的选择

---

### 创新点 10：GroupNorm + Timestep 加法注入条件（vs AdaLN-Zero 全局调制）

**FLUX2 的做法**：
- 使用**全局共享的 AdaLN-Zero 调制**：时间步和 guidance 嵌入通过 `Modulation` 模块生成 shift/scale/gate 参数，对每层的 LayerNorm 输出进行仿射变换
- 所有 DoubleStreamBlock 共享一组 img/txt modulation，所有 SingleStreamBlock 共享一组 modulation
- 代码证据（`model.py` L98-108, L400-412, L469-471）：
  ```python
  self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, disable_bias=True)
  self.single_stream_modulation = Modulation(self.hidden_size, double=False, disable_bias=True)
  # forward:
  x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
  output = x + mod_gate * output
  ```

**DreamLite 的做法**：
- 时间步嵌入通过**加法注入**到 ResnetBlock2D 的中间特征图：时间步经 sinusoidal embedding → MLP 后，直接加到卷积层输出上
- 使用 **GroupNorm** 进行特征归一化（而非 LayerNorm）
- 代码证据（`resnet.py` L288, L297-301, L374-381）：
  ```python
  self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
  self.time_emb_proj = nn.Linear(temb_channels, out_channels)
  # forward:
  temb = self.time_emb_proj(temb)[:, :, None, None]
  hidden_states = hidden_states + temb  # 加法注入
  hidden_states = self.norm2(hidden_states)
  ```

**优点与分析**：
- **GroupNorm vs LayerNorm**：GroupNorm 更适合卷积架构，因为它在通道维度分组归一化，不受空间维度影响；LayerNorm 更适合序列型 Transformer
- **加法注入 vs AdaLN-Zero**：加法注入更简单直接，计算开销更小；AdaLN-Zero 通过 scale/shift/gate 提供更强的条件调制能力
- FLUX2 的全局共享 Modulation 节省了参数（3 个 Modulation 替代每层独立的），DreamLite 的时间步条件注入是每个 ResBlock 独立的
- 优化维度为**实现简洁性和计算效率**

---

### 创新点 11：多种特殊 Block 变体提供的架构灵活性

**FLUX2 的做法**：
- 只有两种 block 类型：`DoubleStreamBlock` 和 `SingleStreamBlock`，架构固定

**DreamLite 的做法**：
- 提供丰富的 block 变体用于灵活组合：
  - `CrossAttnDownRemoveSelfAttnBlock2D`：移除 Self-Attention 的下采样 Cross-Attention 块
  - `CrossAttnUpRemoveSelfAttnBlock2D`：移除 Self-Attention 的上采样 Cross-Attention 块
  - `CrossAttnUpRemoveSelfAttnBlock2DV1`：上采样在 ResBlock+Transformer 之后（不同于默认在最后）
  - `UNetMidBlock2DCrossAttnWoResBlock`：无 ResBlock 的中间层（仅 Cross-Attention Transformer）
  - `UNetMidBlock2D`：纯 ResBlock 中间层（`add_attention=False` 时无注意力）
  - `UpBlockUpSampleFirst2D`：上采样操作放在 block 最前面
  - `DownBlock2D` / `UpBlock2D`：纯 ResBlock，无注意力
- 代码证据（`unet_2d_blocks.py` L43-710）中的 `get_down_block`、`get_mid_block`、`get_up_block` 工厂函数

**优点与分析**：
- 提供了极大的**架构搜索空间**：可以在不同层级选择不同的 block 类型，例如低分辨率层使用 Self-Attention，高分辨率层移除 Self-Attention 以节省计算
- 支持非对称 UNet 设计（通过 `reverse_transformer_layers_per_block` 参数）
- 优化维度为**架构设计灵活性**，有助于找到最佳的精度-效率权衡

---

### 创新点 12：显式的 Encoder Attention Mask 支持

**FLUX2 的做法**：
- 文本 tokens 通过位置 ID（`ctx_ids`）区分，但在注意力计算中不使用显式的 attention mask
- 代码证据（`model.py` L758-815）中的 `causal_attn_fn` 不接受 mask 参数（除了因果注意力的位置切分）

**DreamLite 的做法**：
- 支持 `encoder_attention_mask`，将变长文本序列的 padding 位置转换为大负值 bias，在 Cross-Attention 中屏蔽无效 token
- 代码证据（`unet_2d_condition_mobile.py` L1207-1209）：
  ```python
  if encoder_attention_mask is not None:
      encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
      encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
  ```
  该 mask 随后传入每个 Cross-Attention 层

**优点与分析**：
- 确保模型不会 attend 到 padding token，避免无效信息干扰
- 在使用 VLM（如 Qwen3VL）编码变长文本+图像序列时尤为重要，因为不同样本的有效 token 长度差异较大
- **可能提升文本-图像对齐质量**，尤其在 batch 内样本文本长度差异较大时

---

### 创新点 13：QK Norm 多种归一化选项

**FLUX2 的做法**：
- 固定使用 **RMSNorm**（自定义带可学习 scale 参数的版本）对 Q 和 K 进行归一化
- 代码证据（`model.py` L734-755）：
  ```python
  class RMSNorm(torch.nn.Module):
      def __init__(self, dim: int):
          self.scale = nn.Parameter(torch.ones(dim))
      def forward(self, x):
          rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
          return (x * rrms).to(dtype=x_dtype) * self.scale

  class QKNorm(torch.nn.Module):
      def __init__(self, dim):
          self.query_norm = RMSNorm(dim)
          self.key_norm = RMSNorm(dim)
  ```

**DreamLite 的做法**：
- 支持多种 QK Norm 类型，通过 `qk_norm` 参数选择：
  - `None`：不使用
  - `"layer_norm"`：标准 LayerNorm
  - `"fp32_layer_norm"`：FP32 精度 LayerNorm
  - `"rms_norm"`：RMSNorm
  - `"rms_norm_across_heads"`：跨头 RMSNorm
  - `"layer_norm_across_heads"`：跨头 LayerNorm
  - `"l2"`：L2 归一化
- 代码证据（`attention_processor.py` L197-221）

**优点与分析**：
- 提供了更多归一化策略的实验选项，不同的归一化方式可能在不同模型规模和数据分布下表现不同
- 跨头归一化（across_heads）是一种更激进的归一化方式，可能在某些场景下更有效
- 优化维度为**训练稳定性的调优灵活性**

---

### 创新点 14：ResnetBlock2D 中的 expand_ratio 中间通道扩展

**FLUX2 的做法**：
- 不使用 ResBlock（纯 Transformer 架构）

**DreamLite 的做法**：
- 当使用深度可分离卷积（`use_sep_conv=True`）时，`ResnetBlock2D` 的第一个卷积层（`conv1`）通过 `expand_ratio=2` 将中间通道数扩展到 2 倍，然后第二个卷积层（`conv2`）通过 `expand_ratio=1/2` 恢复回原始通道数
- 这类似于 MobileNet 中的 Inverted Bottleneck 结构
- 代码证据（`resnet.py` L289-314）：
  ```python
  if use_sep_conv:
      expand_ratio = 2
      self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3,
                                          stride=1, padding=1, expand_ratio=expand_ratio)
      out_channels = out_channels * expand_ratio
      # ...
      self.conv2 = DepthwiseSeparableConv(out_channels, conv_2d_out_channels,
                                          kernel_size=3, stride=1, padding=1, expand_ratio=1 / expand_ratio)
  ```

**优点与分析**：
- Inverted Bottleneck（先扩展后压缩）在移动端模型（MobileNetV2/V3）中被证明是有效的特征提取策略
- 通过在中间层增加通道数，提升卷积层的表达能力，同时输入输出通道数不变
- 优化维度为**在有限参数量下提升特征表达能力**

---

### 创新点 15：Flow Matching + UNet 的组合

**FLUX2 的做法**：
- Flow Matching + Transformer-only（标准组合，当前主流范式）
- 代码证据（`sampling.py`）：使用欧拉步进 `x = x + (t_prev - t_curr) * v_pred`

**DreamLite 的做法**：
- **Flow Matching + UNet**（创新组合），使用 `FlowMatchEulerDiscreteScheduler`
- Flow Matching 原本常与 DIT/Transformer 搭配使用，DreamLite 将其与 UNet 结合
- 代码证据（`pipeline_dreamlite.py` L31, L346-359）：
  ```python
  from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
  # ...
  mu = calculate_shift(image_seq_len, ...)
  timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, ..., sigmas=sigmas, mu=mu)
  ```

**优点与分析**：
- Flow Matching 相比传统 DDPM 采样具有更直的采样轨迹，理论上可以用更少的步数达到相似质量
- 与 UNet 结合后，可以享受 Flow Matching 的采样效率优势，同时保留 UNet 的空间归纳偏置
- DreamLite 进一步支持 4 步推理（通过步骤蒸馏），验证了这种组合的可行性
- 优化维度为**采样效率**，兼顾质量和速度

---

### 创新点 16：Timestep-aware 动态 Sigma 调度

**FLUX2 的做法**：
- 使用固定的线性调度或基于图像序列长度的 `get_schedule` 函数生成 timesteps
- 代码证据（`sampling.py`）

**DreamLite 的做法**：
- 使用 `calculate_shift` 函数根据图像序列长度动态计算 mu 参数，调节噪声调度的偏移
- 支持自定义 sigmas 序列（默认为 `np.linspace(1.0, 1/num_inference_steps, num_inference_steps)`）
- 代码证据（`pipeline_dreamlite.py` L67-77, L345-359）：
  ```python
  def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.16):
      m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
      b = base_shift - m * base_seq_len
      mu = image_seq_len * m + b
      return mu
  ```

**优点与分析**：
- 根据生成图像的分辨率（序列长度）自适应调整噪声调度，高分辨率图像使用更大的 shift，使采样过程更稳定
- 这种自适应调度在 FLUX1 中已有类似设计，DreamLite 将其适配到 UNet 架构中
- 优化维度为**多分辨率生成的质量稳定性**

---

### 创新点 17：Resolution Bucket 系统（多宽高比分辨率桶）

**FLUX2 的做法**：
- 代码中未定义预设的分辨率桶，图像尺寸由用户直接指定

**DreamLite 的做法**：
- 定义了多组预设分辨率桶（`TARGET_BUCKETS_V54` 含 17 种宽高比，`TARGET_BUCKETS_V765` 含 20 种宽高比）
- 根据输入图像的宽高比自动匹配最接近的分辨率桶
- 代码证据（`pipeline_dreamlite.py` L50-62, L102-106）：
  ```python
  TARGET_BUCKETS_V54 = [
      [1248, 832], [1024, 1024], [896, 1184], [832, 1248], ...
  ]
  def _get_closest_bucket(buckets, w, h):
      target_ar = w / h
      best_bucket = min(buckets, key=lambda b: abs((b[0]/b[1]) - target_ar))
      best_bucket = [int(x * 2) for x in best_bucket]
      return tuple(best_bucket)
  ```

**优点与分析**：
- 避免将不同宽高比的图像强制裁剪/拉伸到正方形，减少构图失真
- 在编辑任务中保持原图宽高比更加重要
- 优化维度为**多宽高比输出的质量**

---

## 四、总结对比表

| 维度 | FLUX2 | DreamLite | 优化维度 |
|------|-------|-----------|---------|
| **整体架构** | Transformer-only (MMDIT+DIT) | UNet (Conv+CrossAttn) | 参数效率、端侧部署 |
| **参数量** | ~32B (Dev) / ~4-9B (Klein) | 0.39B | 模型大小 |
| **文本条件注入** | 联合注意力 (Q/K/V 拼接) | 分离式 Cross-Attention | 计算效率 |
| **时间步条件** | 全局 AdaLN-Zero 调制 | ResBlock 加法注入 + GroupNorm | 实现简洁性 |
| **生成/编辑统一** | 序列拼接 + 因果注意力 | 空间宽度拼接 | 架构统一性 |
| **CFG 方式** | Guidance Embedding | 双重 CFG (三路) | 编辑精细控制 |
| **卷积类型** | 无卷积 | 标准 Conv / 深度可分离 Conv | 计算效率 |
| **注意力** | 标准 MHA | 支持 GQA | 内存效率 |
| **FFN 灵活性** | 固定 mlp_ratio | 可配置 ff_mult | 模型容量调控 |
| **Self-Attention** | 必须 | 可选移除 | 推理速度 |
| **文本精炼** | 无 (单层 Linear) | Token Refiner (多种) | 文本理解质量 |
| **QK Norm** | 固定 RMSNorm | 7 种可选 | 训练调优灵活性 |
| **位置编码** | 4D RoPE | 无显式 (卷积隐式) | - |
| **归一化** | LayerNorm | GroupNorm | 适配卷积架构 |
| **Block 变体** | 2 种 | 10+ 种 | 架构搜索空间 |
| **Attention Mask** | 因果分段 (无显式 mask) | 显式 encoder_attention_mask | 变长文本处理 |
| **采样方式** | Flow Matching + Transformer | Flow Matching + UNet | 采样效率 |
| **分辨率处理** | 用户指定 | 分辨率桶自动匹配 | 多宽高比质量 |
| **ResBlock 设计** | 无 | Inverted Bottleneck (expand_ratio) | 特征表达能力 |

---

## 五、结论

DreamLite 的去噪模型相比 FLUX2 的核心设计哲学是**极致轻量化与端侧部署优先**。其创新点大多围绕两个目标：

1. **效率优化**（GQA、深度可分离卷积、可选移除 Self-Attention、可配置 FFN 乘子、UNet 架构等）—— 使得 0.39B 参数的模型能在移动端实时运行
2. **架构灵活性**（多种 Block 变体、多种 QK Norm、多种 Token Refiner 配置等）—— 提供丰富的架构搜索空间，在极小参数量下找到最优配置

在**生成效果**方面可能有益的创新包括：
- In-Context Spatial Concatenation 利用卷积局部感受野增强编辑时的空间对齐
- 双重 CFG 提供更精细的编辑控制
- Token Refiner（尤其是 text_proj_rms 和多层 refiner）可能提升文本理解
- 显式 Encoder Attention Mask 避免 padding 干扰
- 分辨率桶系统保持原图宽高比

总的来说，DreamLite 并非追求在同等参数量下超越 FLUX2 的生成质量，而是在**参数量缩小约 80 倍**的前提下，通过大量轻量化和架构创新，在端侧场景下实现可用的生成与编辑能力。

---

*本分析文档基于 DreamLite（`/opt/nas/p/zhugechaoran/download/code/DreamLite/`）和 FLUX2（`/opt/nas/p/zhugechaoran/download/code/flux2/`）代码仓库的去噪模型源代码逐行分析完成。分析时间：2026年7月31日。*