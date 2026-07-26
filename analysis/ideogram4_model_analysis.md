# Ideogram 4 模型全面分析报告

> 基于 `ideogram4/` 目录下所有源代码的逐文件深入分析

> **代码实现根目录**：`/root/code/ideogram4/`

---

## 目录

1. [VAE 分析](#1-vae-分析)
2. [Flow Matching DIT 模型分析](#2-flow-matching-dit-模型分析)
3. [具体网络结构与子网络](#3-具体网络结构与子网络)
4. [模型网络结构图](#4-模型网络结构图)
5. [文生图与图像编辑能力](#5-文生图与图像编辑能力)
6. [文生图流程图](#6-文生图流程图)
7. [图像编辑流程图](#7-图像编辑流程图)
8. [相比 FLUX2 模型的创新点与改进点](#8-相比-flux2-模型的创新点与改进点)

---

## 1. VAE 分析

### 结论：Ideogram4 使用了 VAE，且与 FLUX2 模型的 VAE 结构完全一致。

### 详细分析

#### 1.1 代码证据

`autoencoder.py` 文件第1行注释明确标注：

```python
"""Flux2 KL autoencoder."""
```

这直接说明 Ideogram4 使用的 VAE 就是 FLUX2 的 KL 自编码器。

#### 1.2 VAE 参数对比

| 参数 | Ideogram4 (`autoencoder.py`) | FLUX2 (`autoencoder.py`) |
|------|------|------|
| `resolution` | 256 | 256 |
| `in_channels` | 3 | 3 |
| `ch` | 128 | 128 |
| `out_ch` | 3 | 3 |
| `ch_mult` | [1, 2, 4, 4] | [1, 2, 4, 4] |
| `num_res_blocks` | 2 | 2 |
| `z_channels` | 32 | 32 |

参数**完全一致**。

#### 1.3 结构对比

两者的 VAE 结构完全相同，都包含：

- **Encoder**：`conv_in` → 4级下采样（每级含ResnetBlock + 可选AttnBlock + Downsample）→ `mid`（ResnetBlock + AttnBlock + ResnetBlock）→ `norm_out` + `conv_out` → `quant_conv`
- **Decoder**：`post_quant_conv` → `conv_in` → `mid` → 4级上采样（每级含ResnetBlock + 可选AttnBlock + Upsample）→ `norm_out` + `conv_out`
- **AutoEncoder**：包含 `encoder`、`decoder`，以及 `BatchNorm2d`（用于 latent 归一化，`patch_size=[2,2]`）

#### 1.4 与 FLUX1 VAE 的区别

FLUX2 的 VAE 相比 FLUX1 有显著改进（FLUX2 README 中提到 "The FLUX.2 autoencoder has considerably improved over the FLUX.1 autoencoder"）。关键区别在于：

- FLUX2/Ideogram4 的 `z_channels = 32`（FLUX1 为 16）
- FLUX2/Ideogram4 使用 `BatchNorm2d` 进行 latent 归一化（FLUX1 没有）
- FLUX2/Ideogram4 使用 2×2 patchify（将 `z_channels` 从 32 扩展为 128 通道）

**因此，Ideogram4 使用的是 FLUX2 VAE 结构，不是 FLUX1 VAE 结构。**

#### 1.5 Ideogram4 中 VAE 的使用方式

在 `pipeline_ideogram4.py` 中：
- **仅使用了 VAE Decoder**（`self.autoencoder.decoder(z)`），用于将去噪后的 latent 解码为像素图像
- **未使用 VAE Encoder**（没有编码输入图像的代码路径）
- 加载时使用 `convert_diffusers_state_dict` 将 diffusers 格式的权重键名转换为自定义格式

---

## 2. Flow Matching DIT 模型分析

### 结论：Ideogram4 使用了 flow_matching 的 DIT 模型，且是**单流（Single-Stream）DIT 模型**，不是双流 MMDIT。

### 详细分析

#### 2.1 Flow Matching 证据

1. **`modeling_ideogram4.py` 第1行注释**：
   ```python
   """Ideogram4 transformer backbone. The transformer consumes Qwen3-VL embeddings 
   and flow-matching noise tokens to produce velocity predictions on image latents."""
   ```

2. **`Ideogram4Transformer` 类注释**（第266行）：
   ```python
   """Ideogram 4 flow-matching transformer."""
   ```

3. **`scheduler.py` 第1行注释**：
   ```python
   """Logit-normal schedule and Euler flow-matching sampler."""
   ```

4. **Pipeline 中的 Euler 积分**（`pipeline_ideogram4.py` 第587-616行）：
   ```python
   for i in range(num_steps - 1, -1, -1):
       # ...
       v = gw_i * pos_v + (1.0 - gw_i) * neg_v  # velocity prediction
       delta = s_val - t_val
       z = z + v * delta  # Euler step: z_{t-dt} = z_t + v * dt
   ```

   这是标准的 flow-matching Euler 采样器，模型预测的是速度场 `v(z_t, t)`。

#### 2.2 单流 DIT 证据

1. **只有一种 Transformer Block**：`Ideogram4TransformerBlock`（不像 FLUX2 有 `DoubleStreamBlock` 和 `SingleStreamBlock` 两种）

2. **文本和图像 token 拼接成一个序列**（`pipeline_ideogram4.py` 第344-412行 `_build_inputs`）：
   ```
   序列布局：[padding] [text tokens] [image latent tokens]
   ```

3. **共用同一个自注意力层**（`modeling_ideogram4.py` 第363-364行）：
   ```python
   h = x + llm_features  # 文本特征和图像特征直接相加到同一隐藏空间
   ```

4. **官方文档确认**（`docs/model_architecture.md` 第26行）：
   ```
   The transformer is a single-stream DiT: text tokens (Qwen3-VL hidden states from
   the activation layers) and image latent tokens are concatenated into one
   sequence...
   ```

5. **与 FLUX2 的双流+单流混合架构形成对比**：
   - FLUX2：8个 `DoubleStreamBlock`（文本和图像分别有独立的 QKV 投影、MLP 和归一化层，只在注意力计算时拼接）+ 48个 `SingleStreamBlock`
   - Ideogram4：34个 `Ideogram4TransformerBlock`（文本和图像完全共享所有层）

---

## 3. 具体网络结构与子网络

### Ideogram4 共有以下 4 个子网络（组件）：

### 3.1 子网络一：Qwen3-VL-8B-Instruct 文本编码器

| 属性 | 值 |
|------|------|
| 类型 | 冻结的视觉语言模型（仅用文本模式） |
| 模型 | Qwen3-VL-8B-Instruct |
| 隐藏维度 | 4096 |
| 层数 | 36 层 |
| 抽取层 | 13层：(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35) |
| 输出维度 | 4096 × 13 = 53,248 |
| 作用 | 将文本提示编码为多尺度语义特征 |

**代码位置**：`pipeline_ideogram4.py` 中的 `_encode_text` 和 `_get_qwen3_vl_embeddings` 方法

**工作方式**：
1. 使用 Qwen3 chat template 对文本进行 tokenize
2. 前向传播经过 36 层 transformer
3. 从 13 个指定层提取隐藏状态
4. 将这些隐藏状态沿特征维度拼接，得到 `(B, L_text, 53248)` 的特征

### 3.2 子网络二：Conditional Transformer（条件 DIT）

| 属性 | 值 |
|------|------|
| 类 | `Ideogram4Transformer` |
| 嵌入维度 `emb_dim` | 4608 |
| 层数 `num_layers` | 34 |
| 注意力头数 `num_heads` | 18 |
| 头维度 | 256 |
| 中间层维度 `intermediate_size` | 12288 |
| AdaLN 维度 `adanln_dim` | 512 |
| 输入通道数 `in_channels` | 128 |
| LLM 特征维度 | 53,248 (4096 × 13) |
| RoPE theta | 5,000,000 |
| MRoPE section | (24, 20, 20) |
| 位置编码 | 3D MRoPE (temporal, height, width) |

**内部组件**：
- `input_proj`：`nn.Linear(128, 4608)` — 将噪声 latent token 投影到嵌入维度
- `llm_cond_norm` + `llm_cond_proj`：RMSNorm + `nn.Linear(53248, 4608)` — 将 LLM 特征投影到嵌入维度
- `t_embedding`：`Ideogram4EmbedScalar` — 时间步正弦嵌入 + MLP
- `adaln_proj`：`nn.Linear(4608, 512)` — 将时间步嵌入投影到 AdaLN 维度
- `embed_image_indicator`：`nn.Embedding(2, 4608)` — 区分图像 token 和文本 token
- `rotary_emb`：`Ideogram4MRoPE` — 3D 多模态旋转位置编码
- `layers`：34 × `Ideogram4TransformerBlock`，每个包含：
  - `Ideogram4Attention`：QK-RMSNorm + 自注意力（含 segment mask）
  - `Ideogram4MLP`：SwiGLU MLP（`w1`, `w3` → SiLU gate → `w2`）
  - `adaln_modulation`：`nn.Linear(512, 4×4608)` — 生成 scale_msa, gate_msa, scale_mlp, gate_mlp
  - 4个 RMSNorm 层（attention 前后各1个，FFN 前后各1个）
- `final_layer`：`Ideogram4FinalLayer` — LayerNorm + AdaLN scale + Linear 输出

### 3.3 子网络三：Unconditional Transformer（无条件 DIT）

与 Conditional Transformer **结构完全一致**（相同的 `Ideogram4Config`），但是**独立的权重**。

**作用**：在 CFG（Classifier-Free Guidance）的无条件分支中使用，输入为零向量的 LLM 特征 + 纯图像 token。

**代码证据**（`pipeline_ideogram4.py` 第253-256行）：
```python
self.conditional_transformer = conditional_transformer
self.unconditional_transformer = unconditional_transformer
```

### 3.4 子网络四：VAE（KL 自编码器）

| 属性 | 值 |
|------|------|
| 类 | `AutoEncoder` |
| 潜在通道数 `z_channels` | 32 |
| 通道乘子 `ch_mult` | [1, 2, 4, 4] |
| 基础通道数 `ch` | 128 |
| ResBlock 数量 | 每级 2 个 |
| 空间压缩因子 | 8× |
| Patch size | 2×2 |
| 有效潜在维度 | 32 × 4 = 128 通道 |

**Encoder 结构**：
- `conv_in(3, 128)` → 4级下采样 → `mid`（ResnetBlock + AttnBlock + ResnetBlock）→ `norm_out` + `conv_out` + `quant_conv`
- 输出：`(B, 64, H/8, W/8)`（均值+方差）

**Decoder 结构**：
- `post_quant_conv` → `conv_in` → `mid` → 4级上采样 → `norm_out` + `conv_out`
- 输出：`(B, 3, H, W)` RGB 图像

**注意**：在 Ideogram4 推理中，**仅使用 Decoder 部分**。

---

## 4. 模型网络结构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Ideogram 4 模型整体架构                                  │
│                                                                                 │
│  文本提示 (JSON/Plain text)                                                     │
│       │                                                                         │
│       ▼                                                                         │
│  ┌──────────────────────────────────────────┐                                   │
│  │  子网络1: Qwen3-VL-8B-Instruct           │                                   │
│  │  (冻结的视觉语言模型, 仅文本模式)          │                                   │
│  │  • Tokenizer (Chat Template)             │                                   │
│  │  • 36层 Transformer                      │                                   │
│  │  • 抽取13层隐藏状态 → 拼接               │                                   │
│  │  输出: (B, L_text, 53248)                │                                   │
│  └─────────────────┬────────────────────────┘                                   │
│                    │ llm_features                                                │
│                    │                                                             │
│                    ▼                                                             │
│  ┌──────────────────────────────────────────┐     ┌──────────────────────────┐   │
│  │  子网络2: Conditional Transformer (DIT)   │     │  子网络3: Unconditional   │   │
│  │  • llm_cond_proj: 53248 → 4608           │     │  Transformer (DIT)       │   │
│  │  • input_proj: 128 → 4608                │     │  • 相同结构, 独立权重     │   │
│  │  • t_embedding: 时间步嵌入               │     │  • 输入: 零LLM特征       │   │
│  │  • 34 × TransformerBlock                 │     │         + 噪声z          │   │
│  │    - QK-RMSNorm Self-Attention (MRoPE)   │     │  • 输出: neg_v           │   │
│  │    - SwiGLU MLP                          │     └──────────┬───────────────┘   │
│  │    - AdaLN (scale/gate from timestep)    │                │                   │
│  │  • FinalLayer                            │                │                   │
│  │  输出: pos_v (velocity prediction)       │                │                   │
│  └─────────────────┬────────────────────────┘                │                   │
│                    │                                          │                   │
│                    │  Asymmetric CFG:                         │                   │
│                    │  v = gw × pos_v + (1-gw) × neg_v        │                   │
│                    ├──────────────────────────────────────────┘                   │
│                    │                                                             │
│                    ▼                                                             │
│            Euler 积分: z = z + v × Δt                                           │
│            (从 t=1 噪声迭代到 t≈0 去噪)                                         │
│                    │                                                             │
│                    ▼                                                             │
│  ┌──────────────────────────────────────────┐                                   │
│  │  子网络4: VAE Decoder                     │                                   │
│  │  • 反归一化 (latent_scale, latent_shift)  │                                   │
│  │  • Unpatchify: 2×2                       │                                   │
│  │  • post_quant_conv → conv_in → mid       │                                   │
│  │  • 4级上采样 (ResnetBlock + Upsample)     │                                   │
│  │  • norm_out → conv_out                   │                                   │
│  │  输出: (B, 3, H, W) RGB图像              │                                   │
│  └─────────────────┬────────────────────────┘                                   │
│                    │                                                             │
│                    ▼                                                             │
│              PIL.Image 输出                                                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 简化箭头示意图

```
文本提示 ──→ [Qwen3-VL-8B] ──→ llm_features ──→ [Conditional DIT] ──→ pos_v ─┐
                                                                                │ CFG
随机噪声 z ──────────────────────────────→ [Unconditional DIT] ──→ neg_v ──────┤
                                                                                │
                                                              v = gw·pos_v + (1-gw)·neg_v
                                                                                │
                                                              Euler: z = z + v·Δt
                                                                                │
                                                              [VAE Decoder] ──→ 图像
```

---

## 5. 文生图与图像编辑能力

### 5.1 文生图：✅ 可以实现

这是 Ideogram4 的**核心功能和唯一功能**。

**代码证据**：
- `Ideogram4Pipeline.__call__` 方法接收 `prompts` 参数，不接收任何图像输入
- `run_inference.py` 只有 `--prompt` 参数，没有图像输入参数
- Pipeline 类注释（第249行）：`"""Ideogram 4 text-to-image pipeline."""`
- README 标题：Ideogram 4 是 "first open-weight **text-to-image** model"

### 5.2 图像+文本提示进行图像编辑：❌ 不能实现

**代码证据**：

1. **Pipeline 没有图像输入接口**：`__call__` 方法的参数列表中没有任何图像输入参数：
   ```python
   def __call__(self, prompts, *, height, width, num_steps, guidance_scale, 
                guidance_schedule, mu, std, seed, schedule, raise_on_caption_issues)
   ```

2. **没有 VAE Encoder 调用**：`pipeline_ideogram4.py` 中没有调用 `self.autoencoder.encoder()` 的代码，也没有 `encode` 方法

3. **没有参考图像 token 处理**：与 FLUX2 不同，Ideogram4 的 `Ideogram4Transformer.forward()` 没有 `x_seq_concat`（参考图像 token）参数

4. **没有 KV Cache 机制**：FLUX2 有 `forward_kv_extract` 和 `forward_kv_cached` 方法用于图像编辑时缓存参考图像的 KV，Ideogram4 完全没有这些机制

5. **Qwen3-VL 仅用文本模式**：虽然 Qwen3-VL 是视觉语言模型，但在 Ideogram4 中 `_get_qwen3_vl_embeddings` 方法只处理 `token_ids`（文本 token），没有处理图像 token 的逻辑（没有 `pixel_values`）

---

## 6. 文生图流程图

### 6.1 输入数据

| 输入 | 类型 | 说明 |
|------|------|------|
| `prompts` | `str` 或 `list[str]` | 文本提示（推荐使用结构化 JSON 格式） |
| `height` | `int` | 输出图像高度（默认1024，需为16的倍数） |
| `width` | `int` | 输出图像宽度（默认1024，需为16的倍数） |
| `num_steps` | `int` | 采样步数（默认48） |
| `guidance_scale` | `float` | CFG 引导权重（默认7.0） |
| `seed` | `int` | 随机种子 |

### 6.2 详细流程图

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    Ideogram 4 文生图完整流程                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  步骤1: 文本验证                                                        ║
║  ┌────────────────────────────────────────┐                             ║
║  │ CaptionVerifier.verify_raw(prompt)     │                             ║
║  │ 验证 JSON 格式、键序、元素类型等        │                             ║
║  └────────────────┬───────────────────────┘                             ║
║                   │                                                     ║
║  步骤2: 构建输入序列                                                     ║
║  ┌────────────────▼───────────────────────┐                             ║
║  │ _build_inputs(prompts, height, width)  │                             ║
║  │                                        │                             ║
║  │ • Tokenize: Qwen3 chat template        │                             ║
║  │ • 计算网格: grid_h = H/(16), grid_w    │                             ║
║  │ • 计算 num_image_tokens = grid_h×grid_w│                             ║
║  │ • 构建 position_ids (3D: t,h,w)        │                             ║
║  │ • 构建 segment_ids (区分padding/sample) │                             ║
║  │ • 构建 indicator (LLM=3 / IMAGE=2)     │                             ║
║  │                                        │                             ║
║  │ 序列布局:                               │                             ║
║  │ [pad_zeros] [text_tokens] [image_slots]│                             ║
║  │                                        │                             ║
║  │ 输出: token_ids, position_ids,          │                             ║
║  │       segment_ids, indicator            │                             ║
║  └────────────────┬───────────────────────┘                             ║
║                   │                                                     ║
║  步骤3: 文本编码 (Qwen3-VL)                                              ║
║  ┌────────────────▼───────────────────────┐                             ║
║  │ _encode_text(token_ids, pos, indicator)│                             ║
║  │                                        │                             ║
║  │ → _get_qwen3_vl_embeddings()           │                             ║
║  │   • embed_tokens(token_ids)            │                             ║
║  │   • 创建 causal_mask                   │                             ║
║  │   • 计算 rotary position_embeddings    │                             ║
║  │   • 逐层前向传播 36 层                 │                             ║
║  │   • 从层 {0,3,6,9,12,15,18,21,        │                             ║
║  │          24,27,30,33,35} 提取隐藏状态  │                             ║
║  │   • 拼接: (B, L, 4096) × 13           │                             ║
║  │         → (B, L, 53248)               │                             ║
║  │   • 零掩码非文本位置                   │                             ║
║  │                                        │                             ║
║  │ 输出: llm_features (B, L, 53248)       │                             ║
║  └────────────────┬───────────────────────┘                             ║
║                   │                                                     ║
║  步骤4: 初始化噪声                                                       ║
║  ┌────────────────▼───────────────────────┐                             ║
║  │ z = randn(B, num_img_tokens, 128)      │                             ║
║  │ (标准高斯噪声, float32)                 │                             ║
║  └────────────────┬───────────────────────┘                             ║
║                   │                                                     ║
║  步骤5: Euler Flow-Matching 采样循环                                      ║
║  ┌────────────────▼───────────────────────┐                             ║
║  │ for i in range(num_steps-1, -1, -1):   │                             ║
║  │                                        │                             ║
║  │   ┌─ 条件分支 (Conditional DIT) ─────┐ │                             ║
║  │   │ pos_z = [text_z_padding, z]       │ │                             ║
║  │   │ pos_out = cond_transformer(       │ │                             ║
║  │   │   llm_features, pos_z, t,         │ │                             ║
║  │   │   position_ids, segment_ids,      │ │                             ║
║  │   │   indicator)                      │ │                             ║
║  │   │ pos_v = pos_out[:, text_len:]     │ │                             ║
║  │   └───────────────────────────────────┘ │                             ║
║  │                                        │                             ║
║  │   ┌─ 无条件分支 (Unconditional DIT) ─┐ │                             ║
║  │   │ neg_v = uncond_transformer(       │ │                             ║
║  │   │   zeros_llm_features, z, t,       │ │                             ║
║  │   │   img_position_ids, img_seg_ids,  │ │                             ║
║  │   │   img_indicator)                  │ │                             ║
║  │   │ (仅图像token,无文本token)          │ │                             ║
║  │   └───────────────────────────────────┘ │                             ║
║  │                                        │                             ║
║  │   CFG 融合:                             │                             ║
║  │   v = gw[i] × pos_v + (1-gw[i]) × neg_v│                            ║
║  │                                        │                             ║
║  │   Euler 步:                             │                             ║
║  │   z = z + v × (s - t)                  │                             ║
║  │                                        │                             ║
║  └────────────────┬───────────────────────┘                             ║
║                   │                                                     ║
║  步骤6: VAE 解码                                                         ║
║  ┌────────────────▼───────────────────────┐                             ║
║  │ _decode(z, grid_h, grid_w)             │                             ║
║  │                                        │                             ║
║  │ • 反归一化: z = z × scale + shift       │                             ║
║  │   (128维 per-channel scale & shift)    │                             ║
║  │                                        │                             ║
║  │ • Unpatchify:                           │                             ║
║  │   (B, gh×gw, 128)                     │                             ║
║  │   → (B, gh, gw, 2, 2, 32)             │                             ║
║  │   → (B, 32, gh×2, gw×2)               │                             ║
║  │                                        │                             ║
║  │ • VAE Decoder forward:                 │                             ║
║  │   post_quant_conv → conv_in → mid →   │                             ║
║  │   4级上采样 → norm_out → conv_out      │                             ║
║  │                                        │                             ║
║  │ • 后处理:                               │                             ║
║  │   clamp(-1, 1) → rescale [0, 255]     │                             ║
║  │   → uint8 → PIL.Image                 │                             ║
║  │                                        │                             ║
║  │ 输出: list[PIL.Image]                   │                             ║
║  └────────────────────────────────────────┘                             ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 6.3 DIT Forward 详细过程

```
┌─ Ideogram4Transformer.forward() ─────────────────────────────────────────┐
│                                                                          │
│  输入:                                                                    │
│  • llm_features: (B, L, 53248)  — Qwen3-VL 多层特征                     │
│  • x: (B, L, 128)              — 噪声 latent tokens                     │
│  • t: (B,)                     — flow-matching 时间步                    │
│  • position_ids: (B, L, 3)     — 3D 位置 (t, h, w)                      │
│  • segment_ids: (B, L)         — 样本分段 ID                             │
│  • indicator: (B, L)           — token 角色 (LLM=3 / IMAGE=2)           │
│                                                                          │
│  ① 创建掩码:                                                              │
│     llm_token_mask = (indicator == 3)                                    │
│     output_image_mask = (indicator == 2)                                 │
│     llm_features *= llm_token_mask   // 零掩码非文本位置                   │
│     x *= output_image_mask           // 零掩码非图像位置                   │
│                                                                          │
│  ② 投影到嵌入空间:                                                         │
│     x = input_proj(x) * output_image_mask     // (B, L, 4608)           │
│                                                                          │
│  ③ 时间步嵌入 → AdaLN 条件:                                                │
│     t_cond = t_embedding(t)                   // 正弦嵌入 + MLP          │
│     adaln_input = SiLU(adaln_proj(t_cond))    // (B, 1, 512)            │
│                                                                          │
│  ④ LLM 特征投影:                                                          │
│     llm_features = llm_cond_norm(llm_features)                          │
│     llm_features = llm_cond_proj(llm_features) * llm_token_mask         │
│                                        // (B, L, 4608)                   │
│                                                                          │
│  ⑤ 融合:                                                                  │
│     h = x + llm_features              // 文本特征 + 图像特征相加          │
│     h += embed_image_indicator(is_image)  // 加图像指示嵌入               │
│                                                                          │
│  ⑥ 计算 MRoPE:                                                            │
│     cos, sin = rotary_emb(position_ids)   // 3D 多模态旋转位置编码        │
│                                                                          │
│  ⑦ 34层 TransformerBlock:                                                  │
│     for layer in layers:                                                 │
│       ┌─ AdaLN Modulation ──────────────────────────────┐                │
│       │ mod = adaln_modulation(adaln_input)              │                │
│       │ scale_msa, gate_msa, scale_mlp, gate_mlp = chunk │                │
│       │ gate_msa = tanh(gate_msa)                       │                │
│       │ gate_mlp = tanh(gate_mlp)                       │                │
│       │ scale_msa = 1 + scale_msa                       │                │
│       │ scale_mlp = 1 + scale_mlp                       │                │
│       └─────────────────────────────────────────────────┘                │
│       ┌─ Self-Attention ────────────────────────────────┐                │
│       │ x_normed = attention_norm1(h) × scale_msa       │                │
│       │ qkv = qkv_proj(x_normed)                        │                │
│       │ q = norm_q(q); k = norm_k(k)   // QK-RMSNorm   │                │
│       │ q, k = apply_rotary(q, k, cos, sin)  // MRoPE   │                │
│       │ attn_mask = segment-based block-diagonal         │                │
│       │ out = SDPA(q, k, v, attn_mask)                  │                │
│       │ out = o_proj(out)                                │                │
│       │ h = h + gate_msa × attention_norm2(out)          │                │
│       └─────────────────────────────────────────────────┘                │
│       ┌─ SwiGLU MLP ───────────────────────────────────┐                 │
│       │ x_normed = ffn_norm1(h) × scale_mlp             │                │
│       │ mlp_out = w2(SiLU(w1(x_normed)) × w3(x_normed))│                │
│       │ h = h + gate_mlp × ffn_norm2(mlp_out)           │                │
│       └─────────────────────────────────────────────────┘                │
│                                                                          │
│  ⑧ 最终输出层:                                                             │
│     scale = 1 + adaln_modulation(SiLU(adaln_input))                      │
│     out = linear(layernorm(h) × scale)                                   │
│     out = out.to(float32)              // (B, L, 128)                    │
│                                                                          │
│  输出: velocity prediction (B, L, 128)                                    │
│  (仅 indicator==OUTPUT_IMAGE_INDICATOR 位置有意义)                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 图像编辑流程图

### 结论：不适用

Ideogram4 当前开源代码**不支持图像+文本提示进行图像编辑**。

**原因总结**：
1. `Ideogram4Pipeline.__call__` 没有图像输入参数
2. 没有调用 VAE Encoder 编码输入图像的代码
3. `Ideogram4Transformer.forward` 没有参考图像 token 拼接机制
4. 没有 KV Cache 机制（FLUX2 用于编辑场景中缓存参考图像 token）
5. Qwen3-VL 虽然是视觉语言模型，但此处仅用作文本编码器（无 `pixel_values` 输入）

**与 FLUX2 的对比**：FLUX2 支持文生图和图像编辑（单参考图像、多参考图像），通过 `forward_kv_extract` / `forward_kv_cached` 实现参考图像 token 的 KV 缓存，以及 `encode_image_refs` 编码参考图像。Ideogram4 完全没有这些功能。

---

## 8. 相比 FLUX2 模型的创新点与改进点

### 8.1 完全单流 DIT 架构 vs FLUX2 的双流+单流混合架构

| 特性 | Ideogram4 | FLUX2 |
|------|-----------|-------|
| 架构类型 | **纯单流 DIT** | 双流 + 单流混合 |
| 文本/图像处理 | 全程共享所有层 | 前8层双流分离，后48层单流融合 |
| Block 类型 | 34 × `Ideogram4TransformerBlock` | 8 × `DoubleStreamBlock` + 48 × `SingleStreamBlock` |

**创新点**：Ideogram4 选择了更简洁的纯单流架构。文本和图像 token 从一开始就在相同的 attention 层中交互，实现"每层都有深度跨模态交互"，无需双流阶段的渐进融合。

### 8.2 视觉语言模型（VLM）替代传统文本编码器

| 特性 | Ideogram4 | FLUX2 |
|------|-----------|-------|
| 文本编码器 | **Qwen3-VL-8B-Instruct**（视觉语言模型） | Mistral-Small-3.2-24B / Qwen3 (纯文本 LLM) |
| 编码器类型 | VLM (视觉+语言) | LLM (仅语言) |
| 编码器用法 | 仅文本模式 | 仅文本模式 |

**创新点**：使用 VLM 而非纯 LLM 作为文本编码器。虽然当前仅用文本模式，但 VLM 在预训练阶段已经学习了视觉-语言对齐，理论上能提供更好的视觉概念理解能力。

### 8.3 多层隐藏状态抽取（13层 vs 3层）

| 特性 | Ideogram4 | FLUX2 |
|------|-----------|-------|
| 抽取层数 | **13层** (0,3,6,9,12,15,18,21,24,27,30,33,35) | 3层 (Mistral: 10,20,30 / Qwen3: 9,18,27) |
| 特征维度 | 4096 × 13 = **53,248** | Mistral: 5120×3=15,360 / Qwen3: 4096×3=12,288 |
| 多尺度覆盖 | 从第0层到第35层全覆盖 | 仅中间到深层 |

**创新点**：抽取更多层的隐藏状态，覆盖从浅层（表面 token 信息）到深层（深层语义）的完整频谱。这提供了更丰富的多尺度文本表示，让 DIT 能够同时利用低层和高层的语言特征。

### 8.4 3D 多模态旋转位置编码（MRoPE）

| 特性 | Ideogram4 | FLUX2 |
|------|-----------|-------|
| 位置编码 | **3D MRoPE** (temporal, height, width) | 4D RoPE (temporal, height, width, length) |
| 编码方式 | 交错式 MRoPE (`Ideogram4MRoPE`) | 分轴拼接式 (`EmbedND`) |
| MRoPE section | (24, 20, 20) | [32, 32, 32, 32] |
| RoPE theta | 5,000,000 | 2,000 |
| 文本/图像统一 | 共享 3D 位置空间 | 4D 坐标空间 |

**创新点**：
- 使用 **3维** 而非 4维 的位置编码（t, h, w），简化了坐标系统
- 采用 **交错式（interleaved）MRoPE**，将不同轴的频率交错排列到 head_dim 中，而非简单拼接
- `rope_theta = 5,000,000`（远大于 FLUX2 的 2,000），支持更大的位置范围
- 文本 token 使用 1D 位置广播到 3 轴，图像 token 使用 (t=0, h, w) 坐标，两者通过 `IMAGE_POSITION_OFFSET=65536` 隔离

### 8.5 AdaLN 调制机制差异

| 特性 | Ideogram4 | FLUX2 |
|------|-----------|-------|
| 调制参数 | **scale + gate**（4参数：scale_msa, gate_msa, scale_mlp, gate_mlp） | **shift + scale + gate**（6参数/双流，3参数/单流） |
| Gate 激活 | **tanh** | 无额外激活 |
| 调制维度 | 独立的 adanln_dim=**512** | 与 hidden_size 相同 (6144) |
| Norm 结构 | **Pre-Norm + Post-Norm**（attention 前后各1个 RMSNorm） | 仅 Pre-Norm (LayerNorm) |
| Norm 类型 | **RMSNorm** | LayerNorm |

**创新点**：
- **无 shift 参数**：只使用 scale 和 gate，不使用 shift，减少参数量
- **tanh gate**：gate 值通过 tanh 限制在 [-1, 1] 范围，提供更稳定的梯度控制
- **独立的 AdaLN 维度**：使用独立的 512 维（而非与 hidden_size 相同），大幅减少 AdaLN 的参数量
- **双重归一化**：attention 和 MLP 前后各有一个 RMSNorm（Pre-Norm 用于输入调制，Post-Norm 用于输出 gate），这与标准 Pre-Norm 或 Post-Norm 方案不同
- **RMSNorm 替代 LayerNorm**：计算更高效

### 8.6 非对称 CFG（Asymmetric Classifier-Free Guidance）

| 特性 | Ideogram4 | FLUX2 |
|------|-----------|-------|
| CFG 方式 | **双独立模型的非对称 CFG** | 单模型 + guidance embedding / 标准 CFG |
| 无条件模型 | **独立权重的 unconditional_transformer** | 同一模型，零条件或 guidance=1 |
| 无条件序列 | **仅图像 token**（无文本 token） | 完整序列（文本 zero/图像全部） |

**创新点**：
- 使用**两个独立权重的 Transformer**（conditional + unconditional），而非一个模型的条件/无条件两次前向传播
- 无条件分支**仅处理图像 token**，完全去除文本 token，减少了无条件前向的计算量
- FLUX2 的 guidance-distilled 版本使用 `guidance_in` embedding 将引导尺度嵌入模型内部，而 Ideogram4 在模型外部进行 CFG 加权

### 8.7 结构化 JSON 提示词系统

| 特性 | Ideogram4 | FLUX2 |
|------|-----------|-------|
| 提示词格式 | **结构化 JSON** | 自然语言文本 |
| 验证系统 | `CaptionVerifier` 完整验证 | 无 |
| 布局控制 | Bounding box 坐标 | 无内置支持 |
| 颜色控制 | Hex 色板 | 无内置支持 |
| 提示词扩展 | Magic Prompt（LLM 扩展） | Prompt Upsampling（LLM 扩展） |

**创新点**：
- 训练时使用**结构化 JSON caption**而非自然语言描述
- JSON 包含 `high_level_description`、`style_description`（美学、光照、色板等）、`compositional_deconstruction`（背景、元素列表含 bbox、文本内容等）
- 内置 `CaptionVerifier` 验证 JSON 格式、键序、类型约束
- 支持 `bbox` 精确空间布局控制和 `color_palette` 颜色控制

### 8.8 Logit-Normal 噪声调度

| 特性 | Ideogram4 | FLUX2 |
|------|-----------|-------|
| 调度类型 | **Logit-Normal** | 线性 + time-SNR shift |
| 参数 | mu, std, logsnr_min/max | 经验 mu 函数 |
| 分辨率自适应 | `mu += 0.5 × log(pixels / 512²)` | `compute_empirical_mu()` |
| 采样方向 | 从 t=1 到 t≈0（反向Euler） | 从 t=1 到 t=0 |

**创新点**：
- 使用 **Logit-Normal 分布** 映射均匀时间步，通过 `ndtri`（正态逆 CDF）+ `expit`（sigmoid）变换
- 提供 `logsnr_min=-15, logsnr_max=18` 的裁剪范围

### 8.9 更紧凑的模型参数

| 特性 | Ideogram4 | FLUX2 [dev] |
|------|-----------|-------------|
| 总参数量 | **9.3B** | 32B |
| DIT hidden_size | 4608 | 6144 |
| DIT 层数 | 34 | 8+48=56 |
| 注意力头数 | 18 | 48 |
| 头维度 | 256 | 128 |
| MLP 中间维度 | 12288 | ~18432 (hidden×3) |
| 文本编码器 | Qwen3-VL-8B | Mistral-24B |

**创新点**：Ideogram4 在仅 9.3B 参数（不到 FLUX2 的 1/3）的情况下，在多个基准上达到了与 FLUX2 可比甚至更优的性能，体现了更高的参数效率。

### 8.10 QK-RMSNorm

| 特性 | Ideogram4 | FLUX2 |
|------|-----------|-------|
| QK Norm | **QK-RMSNorm**（独立的 norm_q, norm_k） | QKNorm（RMSNorm with learnable scale） |
| 实现 | `Ideogram4RMSNorm` 应用于 q, k | `QKNorm` 类 |

两者都使用了 QK-Norm 技术，但 Ideogram4 使用的是标准 RMSNorm（`F.rms_norm`），FLUX2 使用的是自定义 RMSNorm（手动计算 rsqrt）。

### 8.11 Segment-based Attention Mask

| 特性 | Ideogram4 | FLUX2 |
|------|-----------|-------|
| 注意力掩码 | **Segment-based block-diagonal** | 因果注意力 / 全注意力 |
| 实现 | `segment_ids` 决定哪些 token 互相可见 | `causal_attn_fn` 中的 ref/txt/img 分区 |

**创新点**：使用 segment_ids 构建块对角注意力掩码，同一 segment 内的 token 互相可见，不同 segment 的 token 互不可见。这支持了 packed batch（多个样本的序列打包在一起），提高了训练和推理效率。

### 8.12 创新点完整汇总表

| 编号 | 创新点 | 说明 |
|------|--------|------|
| 1 | 纯单流 DIT | 全程单流，无双流阶段 |
| 2 | VLM 作为文本编码器 | 使用 Qwen3-VL 而非纯 LLM |
| 3 | 13层多尺度特征抽取 | 远多于 FLUX2 的 3层 |
| 4 | 3D 交错式 MRoPE | 不同于 FLUX2 的 4D 拼接式 RoPE |
| 5 | 无 shift 的 AdaLN | 只有 scale + tanh gate |
| 6 | 独立 AdaLN 维度 | 512 vs hidden_size，减少参数 |
| 7 | Pre-Norm + Post-Norm 双重归一化 | attention/MLP 前后各有 RMSNorm |
| 8 | 双模型非对称 CFG | 独立权重的 cond/uncond transformer |
| 9 | 无条件分支仅图像 token | 减少无条件计算量 |
| 10 | 结构化 JSON 提示词 | 训练和推理都用 JSON |
| 11 | CaptionVerifier 验证 | 内置提示词格式验证 |
| 12 | Magic Prompt 系统 | LLM 自动扩展提示词 |
| 13 | Logit-Normal 噪声调度 | 替代线性 time-SNR shift |
| 14 | 更高参数效率 | 9.3B 达到 32B 级别性能 |
| 15 | Segment-based 块对角注意力 | 支持 packed batch |
| 16 | Image Indicator Embedding | 显式嵌入区分图像/文本 token |
| 17 | Per-channel latent normalization | 128维独立 shift/scale 替代 BN |
| 18 | 大 RoPE theta (5M) | 支持更大位置范围 |

---

## 附录：关键文件与类对照表

| 文件 | 关键类/函数 | 作用 |
|------|------------|------|
| `modeling_ideogram4.py` | `Ideogram4Transformer`, `Ideogram4TransformerBlock`, `Ideogram4Attention`, `Ideogram4MLP`, `Ideogram4MRoPE`, `Ideogram4FinalLayer`, `Ideogram4EmbedScalar` | DIT 核心模型 |
| `autoencoder.py` | `AutoEncoder`, `Encoder`, `Decoder`, `ResnetBlock`, `AttnBlock` | VAE 自编码器 |
| `pipeline_ideogram4.py` | `Ideogram4Pipeline`, `Ideogram4PipelineConfig` | 端到端推理管线 |
| `scheduler.py` | `LogitNormalSchedule`, `SamplerParameters` | 噪声调度器 |
| `constants.py` | `QWEN3_VL_ACTIVATION_LAYERS`, `LLM_TOKEN_INDICATOR`, `OUTPUT_IMAGE_INDICATOR` | 常量定义 |
| `latent_norm.py` | `LATENT_SHIFT`, `LATENT_SCALE` | latent 归一化参数 |
| `caption_verifier.py` | `CaptionVerifier` | JSON 提示词验证 |
| `magic_prompt.py` | `MagicPrompt`, `ClaudeSonnetMagicPromptV1` | 提示词扩展 |
| `sampler_configs.py` | `PRESETS` | 采样器预设配置 |
| `safety.py` | `moderate_prompt`, `moderate_image` | 安全审核 |
| `quantized_loading.py` | `Fp8Linear`, `swap_linears_to_bnb4bit` | 量化加载 |
