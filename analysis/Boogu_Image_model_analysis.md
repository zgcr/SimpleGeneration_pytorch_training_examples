# Boogu-Image 模型全面分析报告

> 本报告基于对 Boogu-Image 代码库的全面代码分析得出结论，涵盖模型架构、子网络结构、推理流程及与 FLUX2 的对比分析。

> **代码实现根目录**：`/root/code/Boogu-Image/`

---

## 目录

1. [VAE 使用情况分析](#1-vae-使用情况分析)
2. [Flow Matching DIT 模型分析](#2-flow-matching-dit-模型分析)
3. [模型具体网络结构及子网络](#3-模型具体网络结构及子网络)
4. [模型网络结构图](#4-模型网络结构图)
5. [文生图与图像编辑能力分析](#5-文生图与图像编辑能力分析)
6. [文生图推理流程图](#6-文生图推理流程图)
7. [图像编辑推理流程图](#7-图像编辑推理流程图)
8. [相比 FLUX2 的创新点与改进点](#8-相比-flux2-的创新点与改进点)

---

## 1. VAE 使用情况分析

### 结论

**Boogu-Image 使用了 VAE，使用的是与 FLUX1 模型 VAE 相同的结构（diffusers 标准 `AutoencoderKL`，latent 通道数为 16），而非 FLUX2 的 VAE 结构。**

### 代码依据

#### 1.1 VAE 导入与初始化

在 `pipeline_boogu.py` 中：

```python
from diffusers.models.autoencoders import AutoencoderKL

class BooguImagePipeline(DiffusionPipeline, BooguImageLoraLoaderMixin):
    def __init__(
        self,
        transformer: BooguImageTransformer2DModel,
        vae: AutoencoderKL,  # 使用 diffusers 标准 AutoencoderKL
        scheduler: FlowMatchEulerDiscreteScheduler,
        mllm: Qwen3VLForConditionalGeneration,
        processor: Qwen3VLProcessor,
    ) -> None:
        ...
```

#### 1.2 VAE 编码过程

```python
def encode_vae(self, img: torch.FloatTensor) -> torch.FloatTensor:
    z0 = self.vae.encode(img.to(dtype=self.vae.dtype)).latent_dist.sample()
    if self.vae.config.shift_factor is not None:
        z0 = z0 - self.vae.config.shift_factor
    if self.vae.config.scaling_factor is not None:
        z0 = z0 * self.vae.config.scaling_factor
    z0 = z0.to(dtype=self.vae.dtype)
    return z0
```

#### 1.3 VAE 解码过程

```python
# processing 方法末尾
if self.vae.config.scaling_factor is not None:
    latents = latents / self.vae.config.scaling_factor
if self.vae.config.shift_factor is not None:
    latents = latents + self.vae.config.shift_factor
image = self.vae.decode(latents, return_dict=False)[0]
```

#### 1.4 与 FLUX1/FLUX2 VAE 的对比

| 特征 | Boogu-Image VAE | FLUX1 VAE | FLUX2 VAE |
|------|----------------|-----------|-----------|
| 类型 | `AutoencoderKL` (diffusers) | `AutoencoderKL` (diffusers) | 自定义 `AutoEncoder` |
| Latent 通道数 | 16（从 `in_channels=16` 推断） | 16 | 128（32×2×2 patch化后） |
| 编码方式 | `latent_dist.sample()` | `latent_dist.sample()` | 取均值 + patch rearrange + BatchNorm |
| 归一化 | shift_factor + scaling_factor | shift_factor + scaling_factor | BatchNorm (running stats) |
| 下采样倍数 | 8× | 8× | 8×（但因 patch 化，有效为 16×） |

**结论：Boogu-Image 使用的 VAE 结构与 FLUX1 模型的 VAE 一致。**

---

## 2. Flow Matching DIT 模型分析

### 结论

**Boogu-Image 使用了 Flow Matching 的 DIT 模型。它采用的是"混合流"架构：先双流 MMDIT 层，再单流 DIT 层，而非纯粹的单流 DIT 或纯粹的双流 MMDIT。**

### 代码依据

#### 2.1 Flow Matching Scheduler

`scheduling_flow_match_euler_discrete_time_shifting.py` 实现了 Flow Matching Euler 调度器：

```python
class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    def step(self, model_output, timestep, sample, ...):
        t = self._timesteps[self.step_index]
        t_next = self._timesteps[self.step_index + 1]
        prev_sample = sample + (t_next - t) * model_output  # Flow Matching Euler Step
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
```

这是标准的 Flow Matching Euler 离散步进公式：`x_{t+1} = x_t + (t_{next} - t) * v_t`。

#### 2.2 混合流 Transformer 架构

`BooguImageTransformer2DModel` 的构造函数中明确了混合架构：

```python
# 双流层（Double-Stream Layers）
self.double_stream_layers = nn.ModuleList([
    BooguImageDoubleStreamTransformerBlock(...)
    for _ in range(num_double_stream_layers)  # 默认值 2
])

# 单流层（Single-Stream Layers）
self.single_stream_layers = nn.ModuleList([
    BooguImageSingleStreamTransformerBlock(...)
    for _ in range(self.num_single_stream_layers)  # = num_layers - num_double_stream_layers = 24
])
```

#### 2.3 前向传播流程

```python
def forward(self, ...):
    # 1. 双流阶段：instruction tokens 和 image tokens 分别处理
    for layer in self.double_stream_layers:
        img_hidden_states, instruct_hidden_states = layer(
            img_hidden_states, instruct_hidden_states, ...)

    # 2. 融合：将两个流合并为一个序列
    joint_hidden_states[i, :encoder_seq_len] = instruct_hidden_states[...]
    joint_hidden_states[i, encoder_seq_len:seq_len] = img_hidden_states[...]

    # 3. 单流阶段：融合后的序列统一处理
    for layer in self.single_stream_layers:
        hidden_states = layer(hidden_states, joint_attention_mask, rotary_emb, temb)

    # 4. 输出投影
    hidden_states = self.norm_out(hidden_states, temb)
```

#### 2.4 双流块内部结构

每个 `BooguImageDoubleStreamTransformerBlock` 包含三种注意力操作：

1. **Joint Attention**（`img_instruct_attn`）：图像和指令 tokens 的联合注意力（跨模态交互）
2. **Image Self-Attention**（`img_self_attn`）：图像 tokens 的独立自注意力
3. **各自的 FFN**：图像流和指令流各自的前馈网络

---

## 3. 模型具体网络结构及子网络

### 子网络结构总览

Boogu-Image 模型由以下 **7 个主要子网络** 组成：

#### 子网络 1：MLLM 指令编码器（Qwen3VL）

- **类型**：`Qwen3VLForConditionalGeneration`（或其 inner model `Qwen3VLModel`）
- **功能**：编码文本指令和可选的参考图像，输出 instruction hidden states
- **输入**：文本 tokens、可选的图像 pixel values
- **输出**：`instruction_embeds` [B, seq_len, hidden_dim]、`instruction_attention_mask` [B, seq_len]
- **特点**：支持多层特征提取（`num_instruction_feat_layers`），支持 concat 或 mean 降维

#### 子网络 2：可选 Prompt Embedding 模块

- **类型**：`PromptEmbedding`
- **功能**：可学习的 prompt tuning tokens，前置到 MLLM 输入嵌入中
- **结构**：
  - `nn.Embedding(num_trainable_prompt_tokens, hidden_size)` — 可训练 token 嵌入
  - `BooguImagePromptTuningRotaryPosEmbed` — Prompt tokens 的 RoPE
  - `BooguImageTransformerBlock` × `num_layers` — 带自注意力和 FFN 的 Transformer 层
- **输入**：token indices、batch_size
- **输出**：`trainable_prompt_embeds` [B, num_tokens, hidden_dim]

#### 子网络 3：VAE Encoder

- **类型**：`AutoencoderKL.encode`（diffusers 标准）
- **功能**：将 RGB 图像编码到 latent 空间
- **输入**：RGB 图像 [B, 3, H, W]
- **输出**：latent 表示 [B, 16, H/8, W/8]
- **后处理**：应用 shift_factor 和 scaling_factor

#### 子网络 4：DIT Transformer（`BooguImageTransformer2DModel`）

这是模型的核心去噪网络，内部包含多个子模块：

##### 4a. 时间步+指令嵌入（`Lumina2CombinedTimestepCaptionEmbedding`）
- **功能**：将时间步编码和指令特征转换为条件嵌入
- **组成**：
  - `Timesteps` → `TimestepEmbedding` → `temb` [B, min(hidden_size, 1024)]
  - `RMSNorm` → `Linear` → `caption_embed` [B, seq_len, hidden_size]

##### 4b. Patch 嵌入层
- `x_embedder`：`nn.Linear(patch_size² × in_channels, hidden_size)` — 噪声图像 patch 嵌入
- `ref_image_patch_embedder`：`nn.Linear(patch_size² × in_channels, hidden_size)` — 参考图像 patch 嵌入
- `image_index_embedding`：`nn.Parameter(5, hidden_size)` — 支持最多 5 张参考图像的索引嵌入

##### 4c. Context Refiner（上下文精炼层）
- **类型**：`BooguImageContextRefinerTransformerBlock` × `num_refiner_layers`
- **功能**：对 instruction hidden states 进行自注意力精炼
- **特点**：`modulation=False`，不使用时间步调制

##### 4d. Noise Refiner（噪声精炼层）
- **类型**：`BooguImageNoiseRefinerTransformerBlock` × `num_refiner_layers`
- **功能**：对噪声图像 latent 的 patch embedding 进行精炼
- **特点**：`modulation=True`，使用时间步调制

##### 4e. Reference Image Refiner（参考图像精炼层）
- **类型**：`BooguImageRefImgRefinerTransformerBlock` × `num_refiner_layers`
- **功能**：对参考图像 latent 的 patch embedding 进行精炼
- **特点**：`modulation=True`，使用时间步调制；参考图像被展平成独立 batch 处理

##### 4f. Double-Stream Layers（双流注意力层）
- **类型**：`BooguImageDoubleStreamTransformerBlock` × `num_double_stream_layers`
- **功能**：图像流和指令流的双流交互处理
- **包含**：
  - `img_instruct_attn`：Joint Attention（使用 `BooguImageDoubleStreamSelfAttnProcessor`，拥有独立的 img_to_q/k/v、instruct_to_q/k/v、img_out、instruct_out 投影层）
  - `img_self_attn`：图像流独立的自注意力
  - `img_feed_forward` / `instruct_feed_forward`：各自的 SwiGLU FFN
  - 多组 RMSNorm 和 `LuminaRMSNormZero` 调制层

##### 4g. Single-Stream Layers（单流注意力层）
- **类型**：`BooguImageSingleStreamTransformerBlock` × `num_single_stream_layers`
- **功能**：对融合后的 [instruction + image] 序列进行统一处理
- **包含**：
  - `attn`：标准自注意力（使用 `BooguImageAttnProcessor`）
  - `feed_forward`：SwiGLU FFN
  - `LuminaRMSNormZero` 调制 + RMSNorm

##### 4h. 输出层（`LuminaLayerNormContinuous`）
- **功能**：条件 LayerNorm + 线性投影，将 hidden_size 映射回 patch_size² × out_channels

#### 子网络 5：VAE Decoder

- **类型**：`AutoencoderKL.decode`（diffusers 标准）
- **功能**：将去噪后的 latent 解码回 RGB 图像
- **输入**：latent [B, 16, H/8, W/8]
- **输出**：RGB 图像 [B, 3, H, W]
- **前处理**：反向应用 scaling_factor 和 shift_factor

#### 子网络 6：Flow Matching Scheduler

- **类型**：`FlowMatchEulerDiscreteScheduler`
- **功能**：控制去噪过程的时间步调度
- **特点**：
  - 支持 v1/v2 两种时间偏移策略
  - 支持动态时间偏移（根据 token 数量自适应调整）
  - Euler 步进：`x_{t+1} = x_t + Δt × v_t`

#### 子网络 7：RoPE 位置编码模块

- **类型**：`BooguImageDoubleStreamRotaryPosEmbed` / `BooguImageRotaryPosEmbed`
- **功能**：为文本 tokens、参考图像 tokens、噪声图像 tokens 生成 3D 旋转位置编码
- **特点**：
  - 3 轴维度：`axes_dim_rope = (40, 40, 40)`，合计 120 = hidden_size / num_heads
  - 文本使用 1D 位置，图像使用 2D 空间位置（行 + 列）
  - 使用复数形式 RoPE（Lumina 风格）

### 默认超参数

| 参数 | 默认值 |
|------|--------|
| `patch_size` | 2 |
| `in_channels` | 16 |
| `hidden_size` | 2304 |
| `num_layers` | 26 |
| `num_double_stream_layers` | 2 |
| `num_refiner_layers` | 2 |
| `num_attention_heads` | 24 |
| `num_kv_heads` | 8 |
| `axes_dim_rope` | (40, 40, 40) |
| `instruction_feat_dim` | 1024 |

---

## 4. 模型网络结构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Boogu-Image 整体架构                         │
└─────────────────────────────────────────────────────────────────┘

  文本指令 (instruction)          可选参考图像 (ref_images)
       │                              │
       │                    ┌─────────┴─────────┐
       │                    │                     │
       │              ┌─────▼─────┐         ┌────▼────┐
       │              │  MLLM     │         │  VAE    │
       │              │ (Qwen3VL) │         │ Encoder │
       │              │ 指令编码器  │         │ 编码器   │
       └──────────────►           │         │         │
                      └─────┬─────┘         └────┬────┘
                            │                     │
                  instruction_embeds        ref_latents
                  + attention_mask       [B, 16, H/8, W/8]
                            │                     │
         ┌──────────────────┼─────────────────────┤
         │                  │                     │
         │     ┌────────────▼────────────┐        │
         │     │   Context Refiner       │        │
         │     │ (instruction 自注意力精炼) │        │
         │     └────────────┬────────────┘        │
         │                  │                     │
         │         instruction_embeds             │
         │                  │         ┌───────────┤
         │                  │         │           │
         │                  │   ┌─────▼──────┐  ┌─▼──────────────┐
         │                  │   │ Ref Image  │  │ Noise Refiner  │
  随机噪声 ──────────────────────► Refiner    │  │ (噪声精炼)      │
  latent                    │   │(参考图精炼) │  └─┬──────────────┘
                            │   └─────┬──────┘    │
                            │         │           │
                            │    ref_img_embed  noise_embed
                            │         │           │
                            │         └─────┬─────┘
                            │               │
                            │    combined_img_hidden_states
                            │     (ref_img + noise_img)
                            │               │
                   ┌────────▼───────────────▼────────┐
                   │                                  │
                   │   Double-Stream Layers (×2)      │
                   │   ┌─────────────┬──────────────┐ │
                   │   │ Instruction │   Image      │ │
                   │   │   Stream    │   Stream     │ │
                   │   │             │              │ │
                   │   │  ┌──Joint Attention──┐     │ │
                   │   │  │ (跨模态交互)       │     │ │
                   │   │  └──────────────────┘     │ │
                   │   │             │              │ │
                   │   │             │ img_self_attn│ │
                   │   │             │ (图像自注意力) │ │
                   │   │   FFN       │    FFN       │ │
                   │   └─────────────┴──────────────┘ │
                   └────────────┬─────────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │     Stream 融合      │
                     │ [instruct + image]  │
                     └──────────┬──────────┘
                                │
                   ┌────────────▼─────────────────┐
                   │                               │
                   │  Single-Stream Layers (×24)   │
                   │  ┌─────────────────────────┐  │
                   │  │  Self-Attention (全序列)  │  │
                   │  │  + SwiGLU FFN            │  │
                   │  │  + RMSNormZero 调制       │  │
                   │  └─────────────────────────┘  │
                   └────────────┬──────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │  Output Projection   │
                     │ (LuminaLayerNorm +   │
                     │  Linear)             │
                     └──────────┬──────────┘
                                │
                         noise_prediction
                                │
                     ┌──────────▼──────────┐
                     │  Scheduler Step      │
                     │  (Euler 步进去噪)     │
                     │  x = x + Δt × v     │
                     └──────────┬──────────┘
                                │
                         (迭代 N 步)
                                │
                     ┌──────────▼──────────┐
                     │    VAE Decoder       │
                     │  (latent → image)   │
                     └──────────┬──────────┘
                                │
                          输出图像 (PIL)
```

---

## 5. 文生图与图像编辑能力分析

### 5.1 文生图（Text-to-Image, T2I）

**结论：✅ 支持**

代码依据：
- `inference_simple.py` 中仅传入文本指令即可生成图像：
  ```python
  image = pipe(
      instruction=instruction,
      negative_instruction=negative_instruction,
      height=output_image_height,
      width=output_image_width,
      num_inference_steps=50,
      text_guidance_scale=4.0,
      ...
  ).images[0]
  ```
- Pipeline 通过 `_get_task_type_by_input_images` 自动判断任务类型，无参考图像时返回 `"t2i"`
- T2I 模式下，`ref_latents` 为 `None`，transformer 的 `ref_image_hidden_states=None`

### 5.2 图像+文本提示进行图像编辑（Text+Image-to-Image, TI2I）

**结论：✅ 支持**

代码依据：
- `inference_ti2i_simple.py` 中同时传入参考图像和编辑指令：
  ```python
  image = pipe(
      instruction=instruction,
      input_image_paths=input_image_paths,
      input_images=input_images,
      negative_instruction=negative_instruction,
      text_guidance_scale=4.0,
      image_guidance_scale=1.0,
      ...
  ).images[0]
  ```
- 有参考图像时任务类型为 `"ti2i"`
- 支持多种编辑操作：对象添加/删除/替换、属性修改、背景替换、风格转换等
- 支持独立的文本引导和图像引导控制

---

## 6. 文生图推理流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                   文生图 (T2I) 推理流程                           │
└─────────────────────────────────────────────────────────────────┘

输入数据:
  ├── instruction: str (文本指令，如 "生成一张夕阳照片")
  ├── negative_instruction: str (负面指令，可选，用于 CFG)
  ├── height, width: int (输出图像尺寸)
  └── num_inference_steps: int (去噪步数)

───────────── 步骤 1：指令编码 ─────────────

  instruction ──→ [Processor (Qwen3VLProcessor)]
                       │
                       │ apply_chat_template + tokenize
                       ▼
                  input_ids + attention_mask
                       │
                       ▼
               [MLLM (Qwen3VL)]
                       │
                       │ forward → last_hidden_state
                       ▼
              instruction_embeds [B, seq_len, 1024]
              instruction_attention_mask [B, seq_len]

  (若启用 CFG) negative_instruction ──→ 同样流程
                       ▼
              negative_instruction_embeds

───────────── 步骤 2：准备 latent ─────────────

  random noise ──→ latents [B, 16, H/8, W/8]
                    (torch.randn)

  ref_latents = None  (T2I 模式无参考图像)

───────────── 步骤 3：计算 RoPE ─────────────

  [BooguImageRotaryPosEmbed] ──→ freqs_cis
                                  (3轴旋转位置编码)

───────────── 步骤 4：迭代去噪 ─────────────

  for t in timesteps:  (共 N 步，如 50 步)
      │
      │ ┌───── Transformer Forward ─────┐
      │ │                                │
      │ │ 1. time_caption_embed:         │
      │ │    timestep → temb             │
      │ │    instruction → caption_embed │
      │ │                                │
      │ │ 2. flat_and_pad_to_seq:        │
      │ │    latent → patch tokens       │
      │ │    (patchify: 2×2 patches)     │
      │ │                                │
      │ │ 3. Context Refiner:            │
      │ │    caption_embed → refined     │
      │ │    (自注意力, 无调制)            │
      │ │                                │
      │ │ 4. Noise Refiner:              │
      │ │    noise patches → refined     │
      │ │    (自注意力 + temb 调制)        │
      │ │                                │
      │ │ 5. Double-Stream Layers (×2):  │
      │ │    img + instruct 双流交互      │
      │ │    (joint attn + self attn)    │
      │ │                                │
      │ │ 6. 融合为 joint_hidden_states  │
      │ │                                │
      │ │ 7. Single-Stream Layers (×24): │
      │ │    统一自注意力处理              │
      │ │                                │
      │ │ 8. Output Projection:          │
      │ │    norm_out → noise prediction │
      │ │                                │
      │ └───────────┬────────────────────┘
      │             │
      │      noise_prediction (model_pred)
      │             │
      │   ┌─────────▼──────────┐
      │   │ CFG 引导 (若启用):   │
      │   │ model_pred =        │
      │   │   model_pred +      │
      │   │   (scale-1) × delta │
      │   └─────────┬──────────┘
      │             │
      │   ┌─────────▼──────────┐
      │   │ Scheduler Step:     │
      │   │ latents = latents + │
      │   │   Δt × model_pred  │
      │   └─────────┬──────────┘
      │             │
      └─────────────┘

───────────── 步骤 5：VAE 解码 ─────────────

  latents ──→ [反向 scaling + shift]
                       │
                       ▼
              [VAE Decoder]
                       │
                       ▼
              image [B, 3, H, W]
                       │
                       ▼
              [F.interpolate] → 最终尺寸
                       │
                       ▼
              [postprocess] → PIL Image

输出数据:
  └── PIL.Image.Image (生成的图像)

最终输出来自: VAE Decoder 子网络
```

---

## 7. 图像编辑推理流程图

```
┌─────────────────────────────────────────────────────────────────┐
│           图像+文本提示图像编辑 (TI2I) 推理流程                    │
└─────────────────────────────────────────────────────────────────┘

输入数据:
  ├── instruction: str (编辑指令，如 "在右下角加上三个柿子")
  ├── input_images: List[PIL.Image] (参考图像)
  ├── negative_instruction: str (负面指令，可选)
  ├── text_guidance_scale: float (文本引导强度，如 4.0)
  └── image_guidance_scale: float (图像引导强度，如 1.0)

───────────── 步骤 1：指令编码 (含图像理解) ─────────────

  instruction + input_images
       │
       ├── images → [Processor resize/preprocess]
       │                    │
       │                    ▼
       │            vlm_input_pil_images
       │
       ▼
  [Processor apply_chat_template]
       │ 构建多模态聊天模板：
       │ system: "Describe the key features..."
       │ user: [image_1] + [image_2] + "编辑指令"
       │
       ▼
  input_ids + pixel_values + attention_mask
       │
       ▼
  [MLLM (Qwen3VL)] ← 理解图像内容 + 编辑指令
       │
       ▼
  instruction_embeds [B, seq_len, 1024]
  instruction_attention_mask [B, seq_len]

  (CFG) negative_instruction ──→ negative_instruction_embeds
  (Double Guide) empty_instruction ──→ empty_instruction_embeds

───────────── 步骤 2：参考图像编码 ─────────────

  input_images ──→ [Processor resize/preprocess]
                       │
                       ▼
                  img_tensor [B, 3, H, W]
                       │
                       ▼
                  [VAE Encoder]
                       │
                       ▼
                  ref_latents [B, 16, H/8, W/8]
                  (每个样本可有多张参考图)

───────────── 步骤 3：准备噪声 latent ─────────────

  random noise ──→ latents [B, 16, H/8, W/8]
                    (尺寸可对齐参考图像)

───────────── 步骤 4：计算 RoPE ─────────────

  [BooguImageDoubleStreamRotaryPosEmbed] ──→
      cap_freqs_cis (文本位置编码)
      ref_img_freqs_cis (参考图位置编码)
      img_freqs_cis (噪声图位置编码)
      combined_img_freqs_cis (ref+noise 联合位置编码)

───────────── 步骤 5：迭代去噪 ─────────────

  for t in timesteps:
      │
      │ ┌───── 条件预测 (model_pred) ─────┐
      │ │ Transformer Forward:              │
      │ │                                   │
      │ │ 1. time_caption_embed:            │
      │ │    t → temb                       │
      │ │    instruction_embeds → caption   │
      │ │                                   │
      │ │ 2. flat_and_pad_to_seq:           │
      │ │    noise latent → noise patches   │
      │ │    ref latent → ref patches       │
      │ │                                   │
      │ │ 3. Context Refiner:              │
      │ │    instruction embeds 自注意力精炼  │
      │ │                                   │
      │ │ 4. img_patch_embed_and_refine:    │
      │ │    ├── x_embedder(noise patches)  │
      │ │    ├── ref_image_patch_embedder   │
      │ │    │   + image_index_embedding    │
      │ │    ├── Noise Refiner (×2)         │
      │ │    ├── Ref Image Refiner (×2)     │
      │ │    └── 合并: [ref + noise] tokens │
      │ │                                   │
      │ │ 5. Double-Stream Layers (×2):     │
      │ │    ├── Joint Attn (instruct+img)  │
      │ │    ├── Image Self-Attn            │
      │ │    └── 各自 FFN                    │
      │ │                                   │
      │ │ 6. 融合 → joint_hidden_states     │
      │ │                                   │
      │ │ 7. Single-Stream Layers (×24)     │
      │ │                                   │
      │ │ 8. Output → noise_prediction      │
      │ └───────────┬───────────────────────┘
      │             │
      │   ┌─────────▼──────────────────────────────┐
      │   │ Double Guidance (text_guide + img_guide):│
      │   │                                         │
      │   │ model_pred_drop_text = predict(          │
      │   │   neg_instruct + ref_latents)           │
      │   │                                         │
      │   │ model_pred_drop_all = predict(           │
      │   │   neg_instruct + None)                  │
      │   │                                         │
      │   │ delta_text = model_pred -                │
      │   │              model_pred_drop_text        │
      │   │ delta_image = model_pred_drop_text -     │
      │   │               model_pred_drop_all        │
      │   │                                         │
      │   │ final = model_pred +                     │
      │   │   (text_scale - 1) × delta_text +       │
      │   │   (image_scale - 1) × delta_image       │
      │   │                                         │
      │   │ (可选: BOG 正交化增强)                     │
      │   └─────────┬──────────────────────────────┘
      │             │
      │   ┌─────────▼──────────┐
      │   │ Scheduler Step:     │
      │   │ latents = latents + │
      │   │   Δt × final_pred  │
      │   └─────────┬──────────┘
      │             │
      └─────────────┘

───────────── 步骤 6：VAE 解码 ─────────────

  latents ──→ [反向 scaling + shift]
                       │
                       ▼
              [VAE Decoder]
                       │
                       ▼
              image [B, 3, H, W]
                       │
                       ▼
              [F.interpolate] → 最终尺寸
                       │
                       ▼
              [postprocess] → PIL Image

输出数据:
  └── PIL.Image.Image (编辑后的图像)

最终输出来自: VAE Decoder 子网络
```

---

## 8. 相比 FLUX2 的创新点与改进点

通过对 Boogu-Image 和 FLUX2 代码的详细对比分析，列举以下创新点和改进点：

### 8.1 多模态大语言模型（MLLM）作为指令编码器

| 维度 | FLUX2 | Boogu-Image |
|------|-------|-------------|
| 文本编码器 | 独立文本编码器（context_in_dim=15360，推测为大型文本模型的拼接特征） | Qwen3VL 多模态大语言模型 |
| 图像理解 | 不理解参考图像的语义内容 | MLLM 可以同时理解图像内容和文本指令 |
| 指令形式 | 简单文本描述 | 多模态聊天模板（system prompt + user message + images） |

**创新性**：Boogu-Image 使用多模态 VLM 作为统一的指令编码器，能够同时理解文本指令和输入图像的语义内容，从而实现更精准的编辑控制。这是与 FLUX2 最本质的区别之一。

### 8.2 统一的文生图和图像编辑架构

| 维度 | FLUX2 | Boogu-Image |
|------|-------|-------------|
| T2I | 支持 | 支持 |
| 图像编辑 | 通过 KV Cache 机制将参考图像 tokens 注入注意力 | 通过 Refiner + 双流架构原生支持参考图像 |
| 架构统一性 | 编辑时需要特殊的 `forward_kv_extract` / `forward_kv_cached` | 同一个 `forward` 函数，通过 `ref_image_hidden_states` 参数控制 |

**创新性**：Boogu-Image 在单一统一架构中原生支持 T2I 和 TI2I，无需为编辑任务设计特殊的 KV Cache 路径。

### 8.3 三种 Refiner 子网络

FLUX2 **没有** Refiner 层。Boogu-Image 引入了三种 Refiner：

1. **Context Refiner**：对 instruction embeddings 进行自注意力精炼（无时间步调制）
2. **Noise Refiner**：对噪声 latent patches 进行精炼（有时间步调制）
3. **Reference Image Refiner**：对参考图像 latent patches 进行精炼（有时间步调制），参考图像被展平为独立 batch 处理

**创新性**：通过专门的 Refiner 层在主体 Transformer 处理之前对不同模态的输入进行预处理和质量提升。

### 8.4 双流块中额外的 Image Self-Attention

| 维度 | FLUX2 DoubleStreamBlock | Boogu-Image DoubleStreamTransformerBlock |
|------|------------------------|----------------------------------------|
| Joint Attention | txt+img 联合注意力 | instruct+img 联合注意力 |
| Image Self-Attention | ❌ 没有 | ✅ 有独立的 `img_self_attn` |
| 注意力次数/块 | 1次 | 2次（joint + self） |

**创新性**：在双流块中，除了跨模态的 Joint Attention，图像 tokens 还有独立的自注意力，允许图像特征在不受文本干扰的情况下进行内部交互。

### 8.5 双流块中独立的 QKV 投影和输出投影

| 维度 | FLUX2 | Boogu-Image |
|------|-------|-------------|
| QKV 投影 | 共享的 `qkv` 线性层 | 图像流和指令流各自独立的 `img_to_q/k/v` 和 `instruct_to_q/k/v` |
| 输出投影 | 共享的 `proj` | 独立的 `img_out` 和 `instruct_out` + 共享的 `to_out` |

**创新性**：双流注意力中，图像和指令使用完全独立的 QKV 投影，允许更灵活的跨模态特征交互。

### 8.6 GQA（Grouped Query Attention）

| 维度 | FLUX2 | Boogu-Image |
|------|-------|-------------|
| 注意力类型 | 标准 MHA（`num_heads=48`，所有 QKV 头数相同） | GQA（`num_attention_heads=24`，`num_kv_heads=8`） |
| KV 头比例 | 1:1 | 3:1（Q heads : KV heads） |

**创新性**：使用 Grouped Query Attention 减少 KV 的计算和内存开销，同时保持模型表达能力。

### 8.7 RMSNorm + SwiGLU FFN（Lumina2 风格）

| 维度 | FLUX2 | Boogu-Image |
|------|-------|-------------|
| 归一化 | LayerNorm (无参数) | RMSNorm |
| FFN 激活 | SiLU (x1*sigmoid(x1) * x2) | SwiGLU (silu(x1) * x2，即相同) |
| FFN 结构 | 单个 Linear + Activation + Linear | 三个 Linear：linear_1, linear_3 → swiglu → linear_2 |
| 调制机制 | Modulation (shift, scale, gate) | LuminaRMSNormZero (scale_msa, gate_msa, scale_mlp, gate_mlp) + tanh gate |

**改进**：使用 RMSNorm 相比 LayerNorm 更高效，tanh gating 提供更稳定的训练信号。

### 8.8 可学习的 Prompt Tuning 模块

FLUX2 **没有** Prompt Tuning 支持。Boogu-Image 引入了 `PromptEmbedding` 模块：

- 包含 `nn.Embedding` 可训练 token 嵌入
- 专用的 `BooguImagePromptTuningRotaryPosEmbed` 位置编码
- 多层 `BooguImageTransformerBlock` 进行 prompt token 的自注意力处理
- 输出前置到 MLLM 输入嵌入中，通过 MLLM 传播梯度

**创新性**：支持轻量级微调场景，通过少量可训练参数适配特定任务。

### 8.9 Image Index Embedding

```python
self.image_index_embedding = nn.Parameter(torch.randn(5, hidden_size))
```

FLUX2 **不支持** 多张参考图像。Boogu-Image 通过 `image_index_embedding` 为不同参考图像添加位置标识，支持最多 5 张参考图像输入。

**创新性**：支持多参考图像输入，为多图编辑、风格融合等任务提供基础。

### 8.10 Boosted Orthogonal Guidance (BOG)

Boogu-Image 引入了全新的 **BOG（Boosted Orthogonal Guidance）** 引导方法：

```python
def calculate_boosted_orthogonal_guidance(self, model_pred, model_pred_uncond, ...):
    delta = model_pred - model_pred_uncond
    delta = momentum_state.update(delta)  # 动量滚动平均
    delta = self.bog_norm(delta)          # Newton-Schulz 正交化
    delta_parallel, delta_orthogonal = self._project_matrix(delta, model_pred, dim=...)
    delta_bog = r_wei * (delta_orthogonal_row + mu * delta_parallel_row) + \
                c_wei * (delta_orthogonal_col + mu * delta_parallel_col)
    return delta_bog
```

关键组件：
1. **Momentum Rolling Sum**：对 delta 做动量滚动平均，平滑引导信号
2. **Newton-Schulz 正交化**：使用 5 步 Newton-Schulz 迭代近似 SVD 正交化
3. **行/列双向投影分解**：沿行和列方向分别分解为平行和正交分量
4. **加权组合**：根据矩阵行列比例加权组合正交和平行分量

FLUX2 **没有** 类似的引导增强机制。

### 8.11 Double Guidance（双重引导）机制

| 维度 | FLUX2 | Boogu-Image |
|------|-------|-------------|
| CFG 引导 | 标准 guidance embedding | 分离的 text_guidance_scale + image_guidance_scale |
| 引导公式 | 简单 CFG | `model_pred + (tgs-1)*Δtext + (igs-1)*Δimage` |
| 空指令引导 | ❌ | ✅ `empty_instruction_guidance_scale` |
| CFG 范围 | 全步 | 可配置 `cfg_range=(start, end)` |

**创新性**：支持独立控制文本和图像的引导强度，并支持空指令引导，提供更精细的生成控制。

### 8.12 指令改写系统（Instruction Rewriter）

Boogu-Image 集成了完整的指令改写系统：

- **本地改写**：使用 MLLM 的生成能力改写用户指令
- **远程改写**：通过 DashScope API 调用远程模型改写
- **多步改写**：支持多轮迭代改写
- **自定义 System Prompt**：支持多种改写策略
- **原始+改写合并**：将原始指令和改写后的指令合并

FLUX2 **没有** 类似的指令改写功能。

### 8.13 TeaCache 和 TaylorSeer 加速

Boogu-Image 集成了两种推理加速技术：

1. **TeaCache**：基于 L1 距离的缓存跳过策略，当输入变化小于阈值时复用上一步的残差
2. **TaylorSeer**：基于 Taylor 展开的特征近似，使用导数近似跳过部分层的计算

每种条件预测（cond/uncond/ref/drop_image/empty_instruct）使用独立的缓存状态，避免跨条件污染。

FLUX2 使用 KV Cache 加速参考图像处理，但没有 TeaCache/TaylorSeer 这类通用加速机制。

### 8.14 动态时间偏移调度

```python
# v2 动态时间偏移
m = sqrt(num_tokens) / scaling_factor
t' = t / (m - m*t + t)
```

Boogu-Image 的 Scheduler 支持：
- **v1/v2 两种时间偏移版本**
- **动态模式**：根据图像 token 数量自适应调整时间偏移
- **静态模式**：使用预设的 `seq_len` 计算时间偏移

FLUX2 的时间步处理更简单，不包含此类自适应时间偏移机制。

### 8.15 复数形式 RoPE

| 维度 | FLUX2 | Boogu-Image |
|------|-------|-------------|
| RoPE 形式 | 旋转矩阵形式 (2×2 matrix) | 复数形式（Lumina 风格） |
| 位置编码维度 | 4 轴 `[32, 32, 32, 32]` | 3 轴 `[40, 40, 40]` |
| θ 基频 | 2000 | 10000 |
| 适用范围 | 仅图像+文本 tokens | 分离的 cap/ref_img/noise/combined RoPE |

**改进**：使用复数形式 RoPE 更简洁高效，3 轴编码设计适配文本-行-列的三维位置结构。

### 8.16 VAE 选择差异

| 维度 | FLUX2 | Boogu-Image |
|------|-------|-------------|
| VAE 类型 | 自定义 AutoEncoder + BatchNorm 归一化 | diffusers AutoencoderKL |
| Latent 通道 | 128 (32 × 2×2 patch) | 16 |
| 归一化方式 | BatchNorm (running stats) | shift_factor + scaling_factor |

**差异**：Boogu-Image 选择了更标准的 FLUX1 风格 VAE，降低了 latent 通道数，可能在计算效率上有优势。

### 创新点总结表

| # | 创新/改进点 | FLUX2 | Boogu-Image |
|---|-----------|-------|-------------|
| 1 | MLLM 指令编码器 | ❌ | ✅ Qwen3VL |
| 2 | 统一 T2I + TI2I 架构 | 部分（KV Cache） | ✅ 原生统一 |
| 3 | 三种 Refiner 子网络 | ❌ | ✅ Context/Noise/RefImage |
| 4 | 双流块 Image Self-Attention | ❌ | ✅ |
| 5 | 双流块独立 QKV+输出投影 | ❌ | ✅ |
| 6 | GQA | ❌ | ✅ (24 Q, 8 KV heads) |
| 7 | RMSNorm + tanh gating | ❌ LayerNorm | ✅ |
| 8 | Prompt Tuning 模块 | ❌ | ✅ |
| 9 | 多参考图 Index Embedding | ❌ | ✅ (最多 5 张) |
| 10 | BOG 正交引导 | ❌ | ✅ Newton-Schulz |
| 11 | Double Guidance | ❌ | ✅ text + image + empty |
| 12 | 指令改写系统 | ❌ | ✅ 本地/远程/多步 |
| 13 | TeaCache + TaylorSeer | ❌ | ✅ |
| 14 | 动态时间偏移 | ❌ | ✅ v1/v2 + dynamic |
| 15 | 复数 RoPE + 3 轴编码 | 旋转矩阵 4 轴 | ✅ 复数 3 轴 |
| 16 | 标准 FLUX1 VAE | 自定义 128ch VAE | ✅ 16ch AutoencoderKL |

---

> **注**：以上分析完全基于代码实现，不涉及模型性能评测。实际效果受训练数据、超参数、训练策略等多方面因素影响。
