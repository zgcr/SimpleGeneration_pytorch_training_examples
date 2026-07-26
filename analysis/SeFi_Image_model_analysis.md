# SeFi-Image 模型全面分析报告

> **代码实现根目录**：`/root/code/SeFi-Image/`

## 项目概述

SeFi-Image（Semantic-First Image）是一个基于**语义优先扩散**（Semantic-First Diffusion）的文生图模型家族。其核心思想是将图像生成过程中的语义（semantic）和纹理（texture）信息分离到两个独立的 latent 通道中，并让语义通道的去噪过程略微领先于纹理通道，从而为纹理生成提供更清晰的结构锚点。

---

## 问题 1：模型是否使用了 VAE？VAE 结构类型？

### 结论：使用了 VAE，但**仅用于纹理（texture）通道**

**详细分析：**

SeFi-Image 模型**使用了 VAE**，但有以下特殊之处：

1. **VAE 仅用于纹理通道的编解码**：在 `TextureLatentCodec` 类（`sefi/modeling/texture_latent_codec.py`）中，VAE 负责将图像编码为纹理 latent 以及将纹理 latent 解码为最终图像。语义通道**不经过任何 VAE**，而是在纯噪声空间中操作。

2. **支持三种 VAE 类型**（见 `sefi/modeling/texture_vae_factory.py` 和 `sefi/modeling/vae_registry.py`）：

   | VAE 名称 | 加载方式 | 结构类型 |
   |---------|---------|---------|
   | `sd1.5` | `diffusers.models.AutoencoderKL.from_pretrained(base_path)` | SD 1.5 标准 VAE |
   | `flux1` | `diffusers.models.AutoencoderKL.from_pretrained(base_path, subfolder="vae")` | FLUX1 VAE（AutoencoderKL） |
   | `flux2` | `diffusers.models.AutoencoderKLFlux2.from_pretrained(base_path, subfolder="vae")` | **FLUX2 VAE（AutoencoderKLFlux2）** |

3. **当使用 `flux2` VAE 时**：
   - VAE 结构与 FLUX2 模型的 VAE **完全一致**（`AutoencoderKLFlux2`）
   - 使用 BatchNorm 进行 latent 归一化（`_use_flux2_bn = True`），通过 `bn.running_mean` 和 `bn.running_var` 进行标准化
   - 归一化公式：`(latent - bn_mean) / bn_std`

4. **当使用 `flux1` 或 `sd1.5` VAE 时**：
   - 使用标准的 `scaling_factor` 和 `shift_factor` 进行 latent 归一化
   - 归一化公式：`(latent - shift_factor) * scaling_factor`

5. **纹理通道维度计算**：`texture_channels = latent_channels * 4`（因为 patchify 操作将 2×2 patch 展平到通道维度）

**关键代码证据：**

```python
# texture_latent_codec.py
class TextureLatentCodec(nn.Module):
    def __init__(self, texture_vae, texture_vae_name):
        self.texture_vae = texture_vae
        self._use_flux2_bn = self.texture_vae_name == "flux2"
        self.latent_channels = int(latent_channels)
        self.texture_channels = int(self.latent_channels * 4)

# vae_registry.py
def load_vae_from_path(model_name, base_path, ...):
    if model_name == "flux2":
        return diffusers.models.AutoencoderKLFlux2.from_pretrained(base_path, subfolder="vae")
```

---

## 问题 2：模型是否使用了 Flow Matching 的 DIT 模型？单流还是双流？

### 结论：使用了 Flow Matching 的 DIT 模型，采用**双流 MMDIT + 单流 DIT 混合结构**

**详细分析：**

1. **Flow Matching 调度器**：模型使用 `FlowMatchEulerDiscreteScheduler`（见 `builder.py` 第 254 行），这是标准的 Flow Matching 采样调度器。

2. **DIT 骨干网络**：基于 `Flux2Transformer2DModel`（来自 diffusers 库），包装在 `Flux2SEFITransformer2DModel` 中。

3. **双流 + 单流混合结构**：
   - **双流 transformer blocks**（`self.backbone.transformer_blocks`）：图像流和文本流分别处理，通过 joint attention 交互
   - **单流 transformer blocks**（`self.backbone.single_transformer_blocks`）：将文本和图像 token 拼接后统一处理

4. **Forward 流程**（见 `flux2_sefi_transformer.py`）：
   ```
   双流阶段：encoder_hidden_states, hidden_states = block(hidden_states, encoder_hidden_states, ...)
   拼接：hidden_states = cat([encoder_hidden_states, hidden_states], dim=1)
   单流阶段：hidden_states = block(hidden_states, ...)
   输出：hidden_states[:, num_txt_tokens:, ...] → norm_out → proj_out
   ```

5. **模型规模预设**（见 `builder.py` 的 `SEFI_SCALE_PRESETS`）：

   | 规模 | attention_head_dim | num_heads | 双流层数 | 单流层数 | joint_attention_dim |
   |-----|-------------------|-----------|---------|---------|-------------------|
   | 0.5B | 128 | 12 | 3 | 10 | 6144 |
   | 1B | 128 | 16 | 4 | 12 | 6144 |
   | 2B | 128 | 20 | 4 | 16 | 6144 |
   | 3B | 128 | 22 | 5 | 18 | 7680 |
   | 4B | 128 | 24 | 5 | 20 | 7680 |
   | 5B | 128 | 26 | 6 | 21 | 7680 |
   | 6B | 128 | 28 | 6 | 22 | 7680 |
   | 8B | 128 | 30 | 7 | 24 | 7680 |
   | 9B | 128 | 32 | 8 | 24 | 12288 |

---

## 问题 3：模型的具体网络结构及子网络组成

### 结论：模型由 4 个核心子网络组成

#### 子网络 1：文本编码器（Qwen3VLTextEncoder）

- **位置**：`sefi/modeling/qwen3vl_text_encoder.py`
- **基础模型**：Qwen3-VL 视觉语言模型（仅使用文本塔，视觉部分被删除以节省显存）
- **支持规模**：`qwen3vl_2b`（hidden_size=2048）、`qwen3vl_4b`（2560）、`qwen3vl_8b`（4096）
- **编码方式**：
  1. 将文本构造为 chat 格式消息
  2. tokenize 并送入 Qwen3-VL 模型
  3. 从指定的隐藏层（默认第 9, 18, 27 层）提取 hidden states
  4. 将多层 hidden states 在特征维度上拼接：`output_dim = hidden_size × num_layers`
  5. 例如 qwen3vl_8b + 3层 → `output_dim = 4096 × 3 = 12288`
- **输出**：`(prompt_embeds, text_ids)`，其中 `prompt_embeds` 形状为 `[batch, seq_len, output_dim]`

#### 子网络 2：纹理 VAE 编码器（Texture VAE Encoder）

- **位置**：`sefi/modeling/texture_latent_codec.py` + `sefi/modeling/vae_registry.py`
- **功能**：将 RGB 图像编码为纹理 latent 表示
- **结构**：取决于选择的 VAE 类型（sd1.5 / flux1 / flux2），均为标准 AutoencoderKL 或 AutoencoderKLFlux2 的 encoder 部分
- **编码流程**：`image → VAE.encode() → raw_latent → normalize → patchify → texture_latent`
- **注意**：文生图时不使用编码器，仅在需要编码参考图像时使用

#### 子网络 3：纹理 VAE 解码器（Texture VAE Decoder）

- **位置**：同上
- **功能**：将去噪后的纹理 latent 解码为最终 RGB 图像
- **解码流程**：`texture_latent → unpatchify → denormalize → VAE.decode() → RGB image`
- **注意**：这是推理管线的最后一步，将纹理通道的 latent 解码为可视图像

#### 子网络 4：SEFI 双时间步 DiT（Flux2SEFITransformer2DModel）

- **位置**：`sefi/modeling/flux2_sefi_transformer.py`
- **功能**：核心去噪网络
- **内部组件**：
  1. **SEFIDualTimestepEmbeddings**：双时间步嵌入模块
     - `time_proj`：正弦时间步投影（共享）
     - `semantic_embedder`：语义时间步 MLP（输出 half_dim）
     - `texture_embedder`：纹理时间步 MLP（输出 half_dim）
     - 输出：`concat([sem_emb, tex_emb])`
  2. **Flux2Transformer2DModel 骨干**：
     - `x_embedder`：图像 latent 输入投影
     - `context_embedder`：文本 embedding 投影
     - `pos_embed`：位置编码（RoPE）
     - `transformer_blocks`：双流 transformer blocks
     - `single_transformer_blocks`：单流 transformer blocks
     - `double_stream_modulation_img/txt`：双流调制层
     - `single_stream_modulation`：单流调制层
     - `norm_out` + `proj_out`：输出归一化和投影层

#### 辅助组件：噪声调度器（FlowMatchEulerDiscreteScheduler）

- **功能**：管理 Flow Matching 采样过程的时间步和 sigma 调度
- **来源**：diffusers 库标准组件

---

## 问题 4：由各子网络组成的模型网络结构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SeFi-Image 模型架构                              │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │  文本提示     │                                                   │
│  │  (Prompt)    │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────────────┐                                       │
│  │  子网络1: 文本编码器       │                                       │
│  │  (Qwen3VLTextEncoder)    │                                       │
│  │  Qwen3-VL 文本塔          │                                       │
│  │  提取第9/18/27层hidden    │                                       │
│  │  states并拼接             │                                       │
│  └──────────┬───────────────┘                                       │
│             │ prompt_embeds + text_ids                               │
│             │                                                       │
│             ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐           │
│  │  子网络4: SEFI 双时间步 DiT                            │           │
│  │  (Flux2SEFITransformer2DModel)                       │           │
│  │                                                      │           │
│  │  ┌─────────────────────────────────────┐             │           │
│  │  │  SEFIDualTimestepEmbeddings         │             │           │
│  │  │  timestep_sem ──→ semantic_embedder │             │           │
│  │  │  timestep_tex ──→ texture_embedder  │──→ temb     │           │
│  │  └─────────────────────────────────────┘             │           │
│  │                                                      │           │
│  │  随机噪声                                             │           │
│  │  (semantic + texture)                                │           │
│  │      │                                               │           │
│  │      ▼                                               │           │
│  │  ┌─────────────┐   ┌──────────────────┐             │           │
│  │  │ x_embedder  │   │ context_embedder │             │           │
│  │  │(图像投影)    │   │(文本投影)        │             │           │
│  │  └──────┬──────┘   └────────┬─────────┘             │           │
│  │         │                   │                        │           │
│  │         ▼                   ▼                        │           │
│  │  ┌──────────────────────────────────┐               │           │
│  │  │  双流 Transformer Blocks         │               │           │
│  │  │  (Double-Stream MMDIT)           │               │           │
│  │  │  图像流 ←→ 文本流 (joint attn)    │               │           │
│  │  └──────────────┬───────────────────┘               │           │
│  │                 │ cat([txt, img])                    │           │
│  │                 ▼                                    │           │
│  │  ┌──────────────────────────────────┐               │           │
│  │  │  单流 Transformer Blocks         │               │           │
│  │  │  (Single-Stream DIT)             │               │           │
│  │  │  统一处理 txt+img tokens          │               │           │
│  │  └──────────────┬───────────────────┘               │           │
│  │                 │ 去除 txt tokens                    │           │
│  │                 ▼                                    │           │
│  │  ┌──────────────────────────────────┐               │           │
│  │  │  norm_out + proj_out             │               │           │
│  │  │  输出: velocity 预测              │               │           │
│  │  └──────────────┬───────────────────┘               │           │
│  └─────────────────┼────────────────────────────────────┘           │
│                    │ velocity (semantic + texture)                   │
│                    │                                                 │
│                    ▼                                                 │
│           ┌────────────────────┐                                     │
│           │  Euler 步进更新      │                                     │
│           │  lat_sem += dt_sem  │                                     │
│           │         × vel_sem  │                                     │
│           │  lat_tex += dt_tex  │                                     │
│           │         × vel_tex  │                                     │
│           └────────┬───────────┘                                     │
│                    │ (循环 N 步)                                      │
│                    ▼                                                 │
│           提取纹理通道 latent                                         │
│           latents[:, semantic_channels:]                              │
│                    │                                                 │
│                    ▼                                                 │
│  ┌──────────────────────────────┐                                    │
│  │  子网络3: 纹理 VAE 解码器     │                                    │
│  │  (Texture VAE Decoder)       │                                    │
│  │  unpatchify → denormalize    │                                    │
│  │  → VAE.decode()              │                                    │
│  └──────────────┬───────────────┘                                    │
│                 │                                                     │
│                 ▼                                                     │
│           最终 RGB 图像                                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 问题 5：模型能否实现文生图？能否实现图像编辑？

### 结论

| 功能 | 是否支持 | 说明 |
|------|---------|------|
| **文生图（T2I）** | ✅ **支持** | 这是模型的核心功能，所有推理代码围绕此构建 |
| **图像+文本提示编辑（I2I Edit）** | ❌ **不支持** | 代码中没有任何图像编辑接口 |

**文生图支持的证据：**
- `SEFIInferencePipeline.__call__()` 接受 `prompts` 参数进行文生图
- `SEFIInferenceRunner.generate_batch()` 实现了完整的文生图推理流程
- README 中明确标注 "text-to-image generation"
- 支持 Base、RL、Turbo 三种模型家族

**不支持图像编辑的证据：**
- `SEFIInferencePipeline.__call__()` 没有任何图像输入参数
- `SEFIInferenceRunner.generate_batch()` 不接受参考图像输入
- `_prepare_latents()` 方法只生成**纯随机噪声**作为初始 latent，没有将输入图像编码为 latent 的流程
- 虽然 `TextureLatentCodec` 有 `encode_texture()` 方法，但该方法在推理管线中**从未被调用**
- 文本编码器 `Qwen3VLTextEncoder` 虽然基于 Qwen3-VL（视觉语言模型），但其**视觉部分在初始化时被删除**（`del self.model.model.visual`），仅保留文本塔
- 整个 CLI 接口（`cli.py`）也没有提供图像输入的选项

---

## 问题 6：文生图流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SeFi-Image 文生图流程                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输入数据:                                                               │
│  ┌────────────────┐  ┌──────────────────────┐                           │
│  │ 文本提示(prompt)│  │ 采样参数              │                           │
│  │ "A red apple   │  │ steps=50             │                           │
│  │  on a table."  │  │ guidance_scale=4.0   │                           │
│  └───────┬────────┘  │ height=1024          │                           │
│          │           │ width=1024           │                           │
│          │           │ seed=42              │                           │
│          │           └──────────────────────┘                           │
│          │                                                              │
│  ════════╪══════════════════════════════════════════════════════════     │
│  步骤1:  │  文本编码                                                     │
│  ════════╪══════════════════════════════════════════════════════════     │
│          ▼                                                              │
│  ┌────────────────────────────────────────────┐                         │
│  │  Qwen3VLTextEncoder.encode(prompts)        │                         │
│  │                                            │                         │
│  │  1. 构建 chat 格式消息                      │                         │
│  │  2. Tokenize (max_length=512, padding)     │                         │
│  │  3. 送入 Qwen3-VL 文本模型                  │                         │
│  │  4. 提取第 9/18/27 层 hidden states        │                         │
│  │  5. stack → permute → reshape              │                         │
│  │     [B, seq_len, hidden_size × 3]          │                         │
│  │  6. 生成 text_ids（4D 坐标）                │                         │
│  └───────────────────┬────────────────────────┘                         │
│                      │                                                  │
│                      ▼                                                  │
│          prompt_embeds: [B, 512, 12288]                                 │
│          text_ids: [B, 512, 4]                                          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════    │
│  步骤2: 准备初始噪声 latent                                              │
│  ════════════════════════════════════════════════════════════════════    │
│                                                                         │
│  ┌────────────────────────────────────────────┐                         │
│  │  _prepare_latents()                        │                         │
│  │                                            │                         │
│  │  latents = torch.randn(                    │                         │
│  │    [B, total_channels, H/2, W/2]           │                         │
│  │  )                                         │                         │
│  │  其中 total_channels =                     │                         │
│  │    semantic_channels + texture_channels     │                         │
│  │  latent_ids = _prepare_latent_ids(latents) │                         │
│  └───────────────────┬────────────────────────┘                         │
│                      │                                                  │
│                      ▼                                                  │
│          latents: [B, total_ch, H/2, W/2]                               │
│          latent_ids: 位置坐标                                            │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════    │
│  步骤3: 构建时间步调度                                                    │
│  ════════════════════════════════════════════════════════════════════    │
│                                                                         │
│  ┌────────────────────────────────────────────┐                         │
│  │  u_base_unit = linspace(0, 1, steps+1)     │                         │
│  │  u_shifted = timestep_shift(u_base, alpha) │                         │
│  │  u_sem_raw = u_shifted × (1 + delta_t)     │                         │
│  │                                            │                         │
│  │  对每一步:                                  │                         │
│  │  u_tex = clamp(u_sem_raw - delta_t, 0, 1)  │                         │
│  │  u_sem = clamp(u_sem_raw, max=1)            │                         │
│  │  → 语义时间步始终 >= 纹理时间步               │                         │
│  └───────────────────┬────────────────────────┘                         │
│                      │                                                  │
│  ════════════════════╪══════════════════════════════════════════════    │
│  步骤4: 迭代去噪循环  │  (共 num_inference_steps 步)                     │
│  ════════════════════╪══════════════════════════════════════════════    │
│                      ▼                                                  │
│  ┌─────────────── 循环开始 ──────────────────────────────────┐          │
│  │                                                           │          │
│  │  4a. 计算当前步的双时间步和 sigma                           │          │
│  │  ┌─────────────────────────────────────────┐              │          │
│  │  │ timesteps_sem, sigmas_sem =              │              │          │
│  │  │   _timesteps_and_sigmas(u_sem)           │              │          │
│  │  │ timesteps_tex, sigmas_tex =              │              │          │
│  │  │   _timesteps_and_sigmas(u_tex)           │              │          │
│  │  └─────────────────────────────────────────┘              │          │
│  │                                                           │          │
│  │  4b. Patchify latents                                     │          │
│  │  ┌─────────────────────────────────────────┐              │          │
│  │  │ packed_latents = _pack_latents(latents)  │              │          │
│  │  └──────────────────┬──────────────────────┘              │          │
│  │                     │                                     │          │
│  │  4c. DiT 前向推理    ▼                                     │          │
│  │  ┌──────────────────────────────────────────────────────┐ │          │
│  │  │  Flux2SEFITransformer2DModel.forward()               │ │          │
│  │  │                                                      │ │          │
│  │  │  输入:                                                │ │          │
│  │  │    hidden_states = packed_latents                     │ │          │
│  │  │    timestep_sem, timestep_tex (双时间步)               │ │          │
│  │  │    encoder_hidden_states = prompt_embeds              │ │          │
│  │  │    txt_ids, img_ids                                   │ │          │
│  │  │                                                      │ │          │
│  │  │  内部计算:                                            │ │          │
│  │  │  1. dual_time_embed(t_sem×1000, t_tex×1000)          │ │          │
│  │  │     → temb = concat([sem_emb, tex_emb])              │ │          │
│  │  │  2. 计算调制参数 (mod_img, mod_txt, mod_single)       │ │          │
│  │  │  3. x_embedder(hidden_states) 图像投影               │ │          │
│  │  │  4. context_embedder(encoder_hidden_states) 文本投影  │ │          │
│  │  │  5. pos_embed 生成 RoPE 位置编码                      │ │          │
│  │  │  6. 双流 transformer blocks (joint attention)        │ │          │
│  │  │  7. cat([txt, img]) 拼接                             │ │          │
│  │  │  8. 单流 transformer blocks                          │ │          │
│  │  │  9. 去除 txt tokens                                  │ │          │
│  │  │  10. norm_out + proj_out                             │ │          │
│  │  │                                                      │ │          │
│  │  │  输出: model_pred (velocity 预测)                     │ │          │
│  │  └──────────────────────┬───────────────────────────────┘ │          │
│  │                         │                                 │          │
│  │  4d. 分离语义和纹理速度 + Euler 步进更新                    │          │
│  │  ┌──────────────────────▼────────────────────────────────┐│          │
│  │  │  vel_sem = velocity[:, :semantic_channels]            ││          │
│  │  │  vel_tex = velocity[:, semantic_channels:]            ││          │
│  │  │  lat_sem = latents[:, :semantic_channels]             ││          │
│  │  │  lat_tex = latents[:, semantic_channels:]             ││          │
│  │  │                                                       ││          │
│  │  │  dt_sem = sigmas_sem_next - sigmas_sem_cur            ││          │
│  │  │  dt_tex = sigmas_tex_next - sigmas_tex_cur            ││          │
│  │  │                                                       ││          │
│  │  │  lat_sem = lat_sem + dt_sem × vel_sem  ← 语义更新     ││          │
│  │  │  lat_tex = lat_tex + dt_tex × vel_tex  ← 纹理更新     ││          │
│  │  │  latents = cat([lat_sem, lat_tex])                    ││          │
│  │  └───────────────────────────────────────────────────────┘│          │
│  │                                                           │          │
│  └─────────────── 循环结束 ──────────────────────────────────┘          │
│                      │                                                  │
│  ════════════════════╪══════════════════════════════════════════════    │
│  步骤5: 解码输出      │                                                  │
│  ════════════════════╪══════════════════════════════════════════════    │
│                      ▼                                                  │
│  ┌────────────────────────────────────────────┐                         │
│  │  提取纹理通道:                              │                         │
│  │  texture_latents =                         │                         │
│  │    latents[:, semantic_channels:]           │                         │
│  │                                            │                         │
│  │  TextureLatentCodec.decode_texture()        │                         │
│  │  1. denormalize (反归一化)                   │                         │
│  │  2. unpatchify (反patch化)                  │                         │
│  │  3. VAE.decode() (VAE 解码器)               │                         │
│  └───────────────────┬────────────────────────┘                         │
│                      │                                                  │
│                      ▼                                                  │
│  ┌────────────────────────────────────────────┐                         │
│  │  Flux2ImageProcessor.postprocess()          │                         │
│  │  tensor → PIL Image                        │                         │
│  └───────────────────┬────────────────────────┘                         │
│                      │                                                  │
│                      ▼                                                  │
│              输出: List[PIL.Image]                                       │
│              最终 RGB 图像                                               │
│              (由 子网络3: 纹理VAE解码器 输出)                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**输入数据总结：**
- 文本提示 (prompt string)
- 采样参数 (steps, guidance_scale, height, width, seed)

**输出数据总结：**
- RGB 图像列表 (List[PIL.Image])
- 最终输出来自**纹理 VAE 解码器**

---

## 问题 7：图像+文本提示进行图像编辑的流程图

### 结论：**不适用**

SeFi-Image 模型**不支持**图像+文本提示的图像编辑功能。

**具体证据：**

1. **推理接口无图像输入**：`SEFIInferencePipeline.__call__()` 仅接受 `prompts` 参数，没有 `image` 参数
2. **初始 latent 全为随机噪声**：`_prepare_latents()` 生成纯随机高斯噪声，不接受任何图像作为初始化
3. **VAE 编码器未被调用**：虽然 `TextureLatentCodec` 有 `encode_texture()` 方法，但在整个推理管线中从未被调用
4. **DiT 无条件图像输入**：`Flux2SEFITransformer2DModel.forward()` 不接受参考图像 token
5. **CLI 无图像参数**：命令行工具没有 `--image` 或类似参数

因此无法画出图像编辑的流程图。模型仅支持纯文生图。

---

## 问题 8：相比 FLUX2 模型的创新点和改进点

### 通过代码对比分析，SeFi-Image 相比 FLUX2 模型具有以下创新点和改进点：

---

### 创新点 1：语义-纹理双通道分离（Semantic-Texture Dual Channel Separation）

**FLUX2**：使用单一 latent 空间（128 通道），所有语义和纹理信息混合在一起。

**SeFi-Image**：将 latent 拆分为**语义通道**和**纹理通道**两部分：
- `total_channels = semantic_channels + texture_channels`
- 语义通道不经过 VAE，在纯噪声空间操作
- 纹理通道由 VAE 编解码，保留高保真细节

```python
# runner.py
vel_sem = velocity[:, :self.semantic_channels]
vel_tex = velocity[:, self.semantic_channels:]
lat_sem = latents[:, :self.semantic_channels]
lat_tex = latents[:, self.semantic_channels:]
```

---

### 创新点 2：语义优先去噪调度（Semantic-First Denoising Schedule）

**FLUX2**：所有 latent 通道使用统一的时间步调度。

**SeFi-Image**：引入 `delta_t` 参数，使语义通道的去噪进度**始终领先于**纹理通道：
- `u_sem_raw = u_shifted × (1 + delta_t)`
- `u_tex = clamp(u_sem_raw - delta_t, 0, 1)`
- `u_sem = clamp(u_sem_raw, max=1)`
- 保证 `u_sem >= u_tex`（语义去噪始终 >= 纹理去噪）

这意味着在去噪过程中，语义结构先被确定，然后纹理细节在已确定的语义结构上生成。

```python
# runner.py - 三阶段掩码去噪
u_sem_raw_schedule = u_shifted_unit * (1.0 + self.delta_t)
u_tex_cur = torch.clamp(u_sem_raw_cur - self.delta_t, min=0.0, max=1.0)
u_sem_cur = torch.clamp(u_sem_raw_cur, max=1.0)
```

---

### 创新点 3：双时间步嵌入（Dual Timestep Embeddings）

**FLUX2**：使用单一时间步嵌入 `time_in(timestep_embedding(t))`，加上可选的 guidance 嵌入 `guidance_in(guidance_embedding(g))`。

**SeFi-Image**：引入 `SEFIDualTimestepEmbeddings` 模块，分别为语义和纹理时间步生成独立的嵌入，然后拼接：

```python
# flux2_sefi_transformer.py
class SEFIDualTimestepEmbeddings(nn.Module):
    def forward(self, timestep_sem, timestep_tex):
        sem_emb = self.semantic_embedder(self.time_proj(timestep_sem))
        tex_emb = self.texture_embedder(self.time_proj(timestep_tex))
        return torch.cat([sem_emb, tex_emb], dim=-1)  # 各占 half_dim
```

同时，FLUX2 的 `time_guidance_embed` 被设置为 `nn.Identity()`，即**移除了 guidance embedding**。

---

### 创新点 4：文本编码器替换（Qwen3-VL 替代 Mistral）

**FLUX2**：使用 `Mistral3SmallEmbedder`（基于 Mistral-Small-3.2-24B-Instruct），提取第 10/20/30 层 hidden states。

**SeFi-Image**：使用 `Qwen3VLTextEncoder`（基于 Qwen3-VL），支持多种规模：
- `qwen3vl_2b`（2B 参数，hidden_size=2048）
- `qwen3vl_4b`（4B 参数，hidden_size=2560）
- `qwen3vl_8b`（8B 参数，hidden_size=4096）
- 提取第 9/18/27 层 hidden states

与 FLUX2 的 24B 文本编码器相比，SeFi-Image 使用更小的文本编码器（最大 8B），且**删除了视觉模块**以节省显存。

---

### 创新点 5：去除 Guidance Embedding

**FLUX2**：在 DiT 中使用 `guidance_in` 模块将 guidance scale 编码为嵌入，加到时间步嵌入上（`vec = vec + guidance_in(guidance_emb)`）。

**SeFi-Image**：
- 构建 transformer 时设置 `guidance_embeds = False`
- 将 backbone 的 `time_guidance_embed` 替换为 `nn.Identity()`
- guidance 完全由双时间步嵌入机制取代

```python
# builder.py
transformer_cfg["guidance_embeds"] = False

# flux2_sefi_transformer.py
self.backbone.time_guidance_embed = nn.Identity()
```

---

### 创新点 6：支持 AutoGuidance 机制

**FLUX2**：使用标准 CFG（Classifier-Free Guidance）或内嵌 guidance embedding。

**SeFi-Image**：除了标准 CFG 外，额外支持 **AutoGuidance**：
- 使用一个**较小的辅助模型**（可以是不同规模的 SeFi 模型）作为无条件预测的基线
- 公式：`velocity = base_pred + scale × (cond_pred - base_pred)`
- 其中 `base_pred` 来自小模型，`cond_pred` 来自大模型
- 可以复用主模型的文本编码器，也可以使用独立的文本编码器

```python
# runner.py
if self.autoguidance_enabled:
    pred_base = self._predict_velocity(self.autoguidance_transformer, ...)
    velocity = _combine_guided_velocity(pred_base, pred_cond, guidance_scale)
```

---

### 创新点 7：Guidance Interval（引导区间限制）

**FLUX2**：guidance 在所有去噪步骤中始终生效。

**SeFi-Image**：支持**限制 guidance 生效的 sigma 区间**：
- 通过 `guidance_interval_sigma_lo` 和 `guidance_interval_sigma_hi` 参数
- 仅当 `sigma_lo < sigma <= sigma_hi` 时才应用 guidance
- 其他时间步直接使用条件预测

```python
# runner.py
def _guidance_interval_is_active(sigma, sigma_lo, sigma_hi):
    return float(sigma_lo) < sigma_value <= float(sigma_hi)
```

---

### 创新点 8：丰富的模型规模预设

**FLUX2**：提供有限的规模选择（主要是完整版 Flux2 和 Klein-4B、Klein-9B）。

**SeFi-Image**：提供**9种模型规模**从 0.5B 到 9B，同时支持自定义规模：
- 0.5B, 1B, 2B, 3B, 4B, 5B, 6B, 8B, 9B
- 支持 `transformer_scale: custom` 并通过 `transformer_overrides` 指定自定义参数

---

### 创新点 9：可配置的时间步偏移调度（Timestep Shift Schedule）

**FLUX2**：使用基于 `generalized_time_snr_shift` 的经验性 mu 计算。

**SeFi-Image**：使用独立的时间步偏移公式：
- `t' = alpha × t / (1 + (alpha-1) × t)`
- 通过 `timestep_shift_alpha` 参数控制
- Base/RL 模型默认 `alpha=0.3`，Turbo 模型默认 `alpha=1.0`

```python
# runner.py
def _apply_timestep_shift_unit_interval(u_unit, alpha):
    return (alpha * u_unit) / (1.0 + (alpha - 1.0) * u_unit)
```

---

### 创新点 10：分离的 Euler 更新步长

**FLUX2**：所有通道使用统一的 `(t_prev - t_curr) * pred` 进行更新。

**SeFi-Image**：语义和纹理通道使用**独立的 dt**：
- `dt_sem = sigmas_sem_next - sigmas_sem_cur`（语义步长）
- `dt_tex = sigmas_tex_next - sigmas_tex_cur`（纹理步长）
- 由于双时间步调度，两者的步长通常不同

```python
# runner.py
lat_sem = lat_sem + dt_sem * vel_sem
lat_tex = lat_tex + dt_tex * vel_tex
```

---

### 创新点 11：仅解码纹理通道

**FLUX2**：去噪后的整个 latent 送入 VAE 解码器。

**SeFi-Image**：去噪完成后，**仅提取纹理通道**送入 VAE 解码器，语义通道被丢弃：

```python
# runner.py
texture_latents = latents[:, self.semantic_channels:]
decoded = self.texture_codec.decode_texture(texture_latents, pipeline_cls=self.pipeline_cls)
```

这体现了模型的核心设计思想：语义通道在去噪过程中提供结构引导，但最终图像仅由纹理通道生成。

---

### 创新点 12：去除参考图像/KV缓存机制

**FLUX2**：支持参考图像输入，通过 `forward_kv_extract` 和 `forward_kv_cached` 实现 KV 缓存加速的图像条件生成。

**SeFi-Image**：**移除了参考图像机制**，DiT 的 forward 只接受标准输入（latent + text embedding + 双时间步），不支持图像条件或 KV 缓存。模型专注于纯文生图任务。

---

### 创新点总结表

| 编号 | 创新点 | FLUX2 | SeFi-Image |
|------|--------|-------|------------|
| 1 | Latent 通道设计 | 单一 128ch latent | 语义+纹理双通道分离 |
| 2 | 去噪调度 | 统一时间步 | 语义优先（delta_t 偏移） |
| 3 | 时间步嵌入 | 单时间步 + guidance 嵌入 | 双时间步嵌入（sem + tex） |
| 4 | 文本编码器 | Mistral-Small-24B | Qwen3-VL（2B/4B/8B） |
| 5 | Guidance 嵌入 | 有（guidance_in） | 无（移除） |
| 6 | 引导方式 | CFG / 内嵌 guidance | CFG + AutoGuidance |
| 7 | Guidance 区间 | 全程生效 | 可限制 sigma 区间 |
| 8 | 模型规模 | 3种规模 | 9种规模 + 自定义 |
| 9 | 时间步偏移 | SNR-based mu | alpha 参数化偏移 |
| 10 | Euler 更新 | 统一 dt | 语义/纹理独立 dt |
| 11 | 解码策略 | 解码全部 latent | 仅解码纹理通道 |
| 12 | 参考图像 | 支持（KV cache） | 不支持（专注 T2I） |
